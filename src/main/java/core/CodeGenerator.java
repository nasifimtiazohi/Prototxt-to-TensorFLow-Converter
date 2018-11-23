package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.PoolingParameter;

public class CodeGenerator {
	private String simpleTensorFlowPython;
    private String multiplexingTensorFlowPython;
    private NetParameter net;
    public List<String> errors;
    private List<String> end_points;
    public CodeGenerator() {
		setSimpleTensorFlowPython("");
		setMultiplexingTensorFlowPython("");
		errors=new ArrayList<String>();
		end_points=new ArrayList<String>();
	}

	public String getSimpleTensorFlowPython() {
		return simpleTensorFlowPython;
	}

	private void setSimpleTensorFlowPython(String simpleTensorFlowPython) {
		this.simpleTensorFlowPython = simpleTensorFlowPython;
	}

	public String getMultiplexingTensorFlowPython() {
		return multiplexingTensorFlowPython;
	}

	private void setMultiplexingTensorFlowPython(String multiplexingTensorFlowPython) {
		this.multiplexingTensorFlowPython = multiplexingTensorFlowPython;
	}

	public void generateNetworkCode(NetParameter netParameter) {
		net=netParameter;
		importStatements();
		networkFunction();
	}

	private void networkFunction() {
		String header=networkFunctionHeader();
		String code=networkFunctionCode();
		code=tabSpaceAllLines(code);
		simpleTensorFlowPython += header + "\n" +code + "\n";
	}

	private String networkFunctionCode() {
		String code="";
		
		String variable_scope_header="with tf.variable_scope(scope,\"Model\",reuse=reuse):";
		String variable_scope_code=variableScope();
		variable_scope_code=tabSpaceAllLines(variable_scope_code);
		code+=variable_scope_header+"\n"+variable_scope_code+"\n";
		
		return code;
	}

	private String variableScope() {
		String code = "";
		
		String arg_scope_header="with slim.arg_scope(default_arg_scope(is_training):";
		String arg_scope_code=argumentScope();
		arg_scope_code=tabSpaceAllLines(arg_scope_code);
		code+=arg_scope_header+"\n"+arg_scope_code+"\n";
		
		return code;
	}

	private String argumentScope() {
		String code="end_points= {}\n";
		
		List<LayerParameter> layers=net.getLayerList();
		
		//assuming one input has been specified
		if (net.getInputCount()==1) {
			code+="end_points['"+net.getInput(0)+"']="+net.getInput(0)+"\n";
		}
		
		for (int i=0;i<layers.size();i++) {
			LayerParameter layer=layers.get(i);
			
			if (layer.getName().contains("/")==false){
				code+=createSimpleLayer(layer);
			}
			else {
				code+=createBranchedlayer(layer);
			}
			
		}
		
		
		return code;
	}
	
	private String createBranchedlayer(LayerParameter layer) {
		String code="";
		if (layer.getType().equals("Input")) {
			//TODO
		}
		else if (layer.getType().equals("Convolution")) {
			code+=createBranchedEndPoint(layer);
		}
		else if (layer.getType().equals("BatchNorm")) {
			//do nothing
		}
		else if (layer.getType().equals("Scale")) {
			//do nothing
		}
		else if (layer.getType().equals("ReLU")) {
			//do nothing
		}
		else if (layer.getType().equals("Pooling")) {
			code+=createBranchedEndPoint(layer);
		}
		return code;
		
	}

	private String createBranchedEndPoint(LayerParameter layer) {
		String[] parts=layer.getName().split("/");
		String root=parts[0];
		
		//check if end_point already exists
		if (end_points.contains(root)) {
			return "";
		}
		else {
			end_points.add(root);
		}
		
		
		String code="end_point='"+root+"'\n";
		
		//recursive code
		code=recursion(root+"/",parts[parts.length-1],code);
		
		
		return code;
	}

	private String recursion(String current, String stop, String code) {
		
		String[] parts=current.split("/"); 
		//TODO handle error cases
		String end_point=parts[parts.length-1];
		
		List<LayerParameter> matched=new ArrayList<LayerParameter>();
		for (int i=0;i<net.getLayerList().size();i++) {
			LayerParameter layer=net.getLayerList().get(i); 
			if (layer.getName().contains(current)){
				matched.add(layer);
			}	
		}
		if (matched.size()>0) {
			String variableScopeHeader="with tf.variable_scope('"+end_point+"'):";
			String variableScopeCode="";
			for (int i=0;i<matched.size();i++) {
				LayerParameter layer=matched.get(i);
				String root=layer.getName().substring(current.length()).split("/")[0];
				root=current+root;
				if (end_points.contains(root)==false) {
					end_points.add(root);
					variableScopeCode=recursion(root+"/", stop, variableScopeCode);
					
				}
			}
			
			//look if the current layer also exists without any branching
			for (int i=0;i<net.getLayerList().size();i++) {
				LayerParameter layer=net.getLayerList().get(i); 
				if (layer.getName().contains(current.subSequence(0, current.length()-1))){
					//work with the layer
					variableScopeCode+=createSimpleLayer(layer);
					break;
				}	
			}
			
			code+=variableScopeHeader+"\n"+tabSpaceAllLines(variableScopeCode)+"\n";
			
		}
		else {
			String parent=parts[parts.length-2];
			String layerName=current.substring(0,current.length()-1);
			LayerParameter layer=null;
			for (int i=0;i<net.getLayerList().size();i++) {
				//TODO inefficient
				if (net.getLayer(i).getName().equals(layerName)) {
					layer=net.getLayer(i);
					break;
				}
			}
			code+=createBranchedEndPoint(layer,parent,end_point);
		}
		return code;
	}


	private String createBranchedEndPoint(LayerParameter layer, String parent, String end_point) {
		String code="";
		if (layer.getType().equals("Convolution")) {
			code+=createBranchedConvolutionEndPoint(layer, parent, end_point);
		}
		else if (layer.getType().equals("BatchNorm")) {
			//do nothing
		}
		else if (layer.getType().equals("Scale")) {
			//do nothing
		}
		else if (layer.getType().equals("ReLU")) {
			//do nothing
		}
		else if (layer.getType().equals("Pooling")) {
			code+=createBranchedPoolingEndPoint(layer, parent, end_point);
		}
		return code;
	}

	private String createSimpleLayer(LayerParameter layer) {
		String code="";
		if (layer.getType().equals("Input")) {
			//TODO
		}
		else if (layer.getType().equals("Convolution")) {
			code+=createSimpleConvolutionEndPoint(layer);
		}
		else if (layer.getType().equals("BatchNorm")) {
			//do nothing
		}
		else if (layer.getType().equals("Scale")) {
			//do nothing
		}
		else if (layer.getType().equals("ReLU")) {
			//do nothing
		}
		else if (layer.getType().equals("Pooling")) {
			code+=createSimplePoolingEndPoint(layer);
		}
		else if(layer.getType().equals("Concat")) {
			//left side
			code+="end_points['"+layer.getName()+"']";
			
			code+="=";
			
			//right side
			code+="tf.concat(axis=3,values=[";
			List<String> bottomList=layer.getBottomList();
			for (int i=0;i<bottomList.size();i++) {
				String name=bottomList.get(i);
				// assuming bottoms would be branches 
				//TODO add error
				name=name.split("/")[1];
				code+=name.toLowerCase();
				if(i<bottomList.size()-1) {
					code+=",";
				}
			}
			code+="])";
		}
		return code;
	}

	private String createBranchedConvolutionEndPoint(LayerParameter layer, String parent,String end_point) {
		
		ConvolutionParameter typeParam=layer.getConvolutionParam();
		
		//left side 
		String code=parent.toLowerCase();
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		String bottom = "end_points['"+layer.getBottom(0)+"']";
		arguments.add(bottom);
		
		//add num_of_output
		arguments.add(Integer.toString(typeParam.getNumOutput()));
		
		//kernel dimension
		//assuming only one kernel size
		if (typeParam.getKernelSizeCount()!=1) {
			errors.add("more than one kernel size for: "+layer.getName());
		}
		String s="[";
		if (typeParam.hasKernelH()) {
			s+=Integer.toString(typeParam.getKernelH());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize(0));
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize(0));
		}
		s+="]";
		arguments.add(s);
		
		//stride
		//assuming single stride
		if (typeParam.getStrideList().size()!=1) {
			errors.add("more than one stride for: "+layer.getName());
		}
		arguments.add("stride="+Integer.toString(typeParam.getStride(0)));
		
		//scope
		arguments.add("scope='"+end_point+"'");
		
		//make slim call
		code+="slim.conv2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		return code;
	}

	private String createSimpleConvolutionEndPoint(LayerParameter layer) {
		//check if end_point already exists
		if (end_points.contains(layer.getName())) {
			return "";
		}
		else {
			end_points.add(layer.getName());
		}
		
		ConvolutionParameter typeParam=layer.getConvolutionParam();
		
		//left side 
		String code="end_points['"+layer.getName()+"']";
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		String bottom = "end_points['"+layer.getBottom(0)+"']";
		arguments.add(bottom);
		
		//add num_of_output
		arguments.add(Integer.toString(typeParam.getNumOutput()));
		
		//kernel dimension
		//assuming only one kernel size
		if (typeParam.getKernelSizeCount()!=1) {
			errors.add("more than one kernel size for: "+layer.getName());
		}
		String s="[";
		if (typeParam.hasKernelH()) {
			s+=Integer.toString(typeParam.getKernelH());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize(0));
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize(0));
		}
		s+="]";
		arguments.add(s);
		
		//stride
		//assuming single stride
		if (typeParam.getStrideList().size()!=1) {
			errors.add("more than one stride for: "+layer.getName());
		}
		arguments.add("stride="+Integer.toString(typeParam.getStride(0)));
		
		//scope
		arguments.add("scope='"+layer.getName()+"'");
		
		//make slim call
		code+="slim.conv2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		return code;
	}

	private String createSimplePoolingEndPoint(LayerParameter layer) {
		//check if end_point already exists
		if (end_points.contains(layer.getName())) {
			return "";
		}
		else {
			end_points.add(layer.getName());
		}
		
		PoolingParameter typeParam=layer.getPoolingParam();
		
		//left side 
		String code="end_points['"+layer.getName()+"']";
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		String bottom = "end_points['"+layer.getBottom(0)+"']";
		arguments.add(bottom);
		
		
		//kernel dimension
		String s="[";
		if (typeParam.hasKernelH()) {
			s+=Integer.toString(typeParam.getKernelH());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize());
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize());
		}
		s+="]";
		arguments.add(s);
		
		//stride
		arguments.add("stride="+Integer.toString(typeParam.getStride()));
		
		//scope
		arguments.add("scope='"+layer.getName()+"'");
		
		//make slim call
		code+="slim.max_pool2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		return code;
	}
	private String createBranchedPoolingEndPoint(LayerParameter layer, String parent,String end_point) {
		PoolingParameter typeParam=layer.getPoolingParam();
		
		//left side 
		String code=parent.toLowerCase();
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		String bottom = "end_points['"+layer.getBottom(0)+"']";
		arguments.add(bottom);
		
		
		//kernel dimension
		String s="[";
		if (typeParam.hasKernelH()) {
			s+=Integer.toString(typeParam.getKernelH());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize());
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			s+=Integer.toString(typeParam.getKernelSize());
		}
		s+="]";
		arguments.add(s);
		
		//stride
		arguments.add("stride="+Integer.toString(typeParam.getStride()));
		
		//scope
		arguments.add("scope='"+end_point+"'");
		
		//make slim call
		code+="slim.max_pool2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		return code;
	}
	private String makeArgumentList(List<String> arguments) {
		String code="(";
		
		for (int i=0;i<arguments.size();i++) {
			if(i<arguments.size()-1){
				code+=arguments.get(i);
				code+=",";
			}
			else {
				//last element
				code+=arguments.get(i);
			}
		}
		
		code+=")";
		
		return code;
	}

	private String makeWithHeader(String name, List<String> arguments, String alias) {
		String code="with "+name;
		code+=makeArgumentList(arguments);
		code+=" as "+alias+":";
		return code;
	}

	private String makeWithHeader(String name, List<String> arguments) {
		String code="with "+name;
		code+=makeArgumentList(arguments);
		code+=":";
		return code;
	}
	private String tabSpaceAllLines(String code) {
		code="\t"+code;
		
		for (int i = -1; (i = code.indexOf("\n", i + 1)) != -1; i++) {
			if (i+1 < code.length()) {
			    String before=code.substring(0, i+1);
			    String after=code.substring(i+1);
			    code=before+'\t'+after;
			}
		} 
		
		return code;
	}

	private String networkFunctionHeader() {
		List<String> arguments=new ArrayList<String>();
		
		String name=net.getName();
		
		String input;
		int input_count=net.getInputCount();
		if (input_count == 1) {
			input=net.getInput(0);
		}
		else {
			input="error: more than one input found";
			errors.add(input);
		}
		arguments.add(input);
		
		String output_count="num_classes=1000"; 
		arguments.add(output_count);
		
		String train_or_test="is_training=true";
		arguments.add(train_or_test);
		
		String variableReuse="reuse=true";
		arguments.add(variableReuse);
		
		
		String scope="scope="+name;
		arguments.add(scope);
		
		String code=makeFunctionHeader(name,arguments);
		
		return code;
	}

	private String makeFunctionHeader(String name, List<String> arguments) {
		String code="def "+name;
		code+=makeArgumentList(arguments);
		code+=":";
		return code;
	}

	private void importStatements() {
		String code="import tensorflow as tf"
				+ "\n"
				+ "slim=tf.contrib.slim" // uses slim api 
				+ "\n\n"
				;
		simpleTensorFlowPython += code;
		
	}

	
	
}
