package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.InnerProductParameter;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.PoolingParameter;

public class MultiplexingCodeGenerator extends CodeGenerator{
	private String multiplexingTensorFlowPython;
	
	public MultiplexingCodeGenerator(){
		setMultiplexingTensorFlowPython("");
		errors=new ArrayList<String>();
		end_points=new ArrayList<String>();
	}
	
	public String getMultiplexingTensorFlowPython() {
		return multiplexingTensorFlowPython;
	}

	private void setMultiplexingTensorFlowPython(String multiplexingTensorFlowPython) {
		this.multiplexingTensorFlowPython = multiplexingTensorFlowPython;
	}
	
	@Override
	public void generateNetworkCode(NetParameter netParameter) {
		
		net=netParameter;
		
		Unhandled_Layers=new ArrayList(net.getLayerList()); //to use iterator remove 
		
		//get the output layer 
		for (int i=net.getLayerList().size()-1;i>=0;i--) {
			LayerParameter layer=net.getLayerList().get(i);
			if (layer.getType().equals("Convolution") || layer.getType().equals("InnerProduct")) {
				outputLayer=layer;
				break;
			}
		}
		
		multiplexingTensorFlowPython+=importStatements();
		multiplexingTensorFlowPython+=networkFunction();
		multiplexingTensorFlowPython+=changeImageSize();
		multiplexingTensorFlowPython+=commonCodeToAllModels();
		
	}
	
	@Override
	protected String networkFunctionHeader() {
		List<String> arguments=new ArrayList<String>();
		
		String name=net.getName();
		
		//Input can be passed as input parameter of the network
		//Or, can be passed a layer of type "input"
		//First, let us search if there is any explicit input parameter
		input = null;
		int input_count=net.getInputCount();
		if (input_count == 1) {
			input=net.getInput(0);
		}
		else if (input_count==0) {
			// Input might be passed as a layer
			List<LayerParameter> layers=net.getLayerList();
			for (int i=0;i<layers.size();i++) {
				LayerParameter layer=layers.get(i);
				if (layer.getType().equals("Input")) {
					input=layer.getName();
					break;
				}
			}
		}
		else{
			input="error: a single input could not be found";
			//Limitation: cannot handle more than one input
			errors.add(input);
		}
		
		arguments.add(input);
		
		//Get the num_output of last output layer (convolution or innerproduct)
		//TODO can get from 
		int output_count=0;
		List<LayerParameter> layers=net.getLayerList();
		
		//get num_output from output layer
		if (outputLayer.getType().equals("Convolution") ) {
			ConvolutionParameter convParam = outputLayer.getConvolutionParam();
			output_count=convParam.getNumOutput();
		}
		else if (outputLayer.getType().equals("InnerProduct")) {
			InnerProductParameter innerProdParam = outputLayer.getInnerProductParam();
			output_count=innerProdParam.getNumOutput();
		}
		
		String num_classes="num_classes="+Integer.toString(output_count);
		arguments.add(num_classes);
		
		//Hardcoded assumption
		String train_or_test="is_training=True";
		arguments.add(train_or_test);
		
		//Hardcoded assumption
		String variableReuse="reuse=None";
		arguments.add(variableReuse);
		
		//Scope is the network name itself
		String scope="scope='"+name+"'";
		arguments.add(scope);
		
		String config="config=None";
		arguments.add(config);
		
		String code=makeFunctionHeader(name,arguments);
		
		return code;
	}

	@Override
	protected String networkFunctionCode() {
		String code="";
		
		// Add template code unique to multiplexing in the beginning
		code+="\n\n############## template code added for multiplexing ##############\n" + 
				"# calculate the number of filter in a conv given config \n" + 
				"selectdepth = lambda k,v: int(config[k]['ratio']*v) if config and k in config and 'ratio' in config[k] else v \n" + 
				"\n" + 
				"# select the input tensor to a module \n" + 
				"selectinput = lambda k, v: config[k]['input'] if config and k in config and 'input' in config[k] else v \n" + 
				"############## end template code ##############"+
				"\n\n";
		
		String variable_scope_header="with tf.variable_scope(scope,\"Model\",reuse=reuse):";
		String variable_scope_code=variableScope();
		variable_scope_code=tabSpaceAllLines(variable_scope_code);
		code+=variable_scope_header+"\n"+variable_scope_code+"\n";
		
		return code;
	}
	
	@Override
	protected Map<String, String> createNewBranch( List<LayerParameter> layers, String code) {
				
		Map<String, String> dictionary = new HashMap<String, String>();
		
		String parent="";
		
		String mixed_scope_header="";
		String mixed_scope_code="";
		
		LayerParameter concatLayer=null;
		
		//check all layers that are branched out
		for (int i=0;i<layers.size();i++) {
			String branch_name= "branch_"+Integer.toString(i);
			String branch_scope_header=makeWithHeader("tf.variable_scope", Arrays.asList("'"+branch_name+"'"))+"\n";
			String branch_scope_code="";
			
			LayerParameter layer=layers.get(i);
			
			if (!layer.getType().equals("Concat")) {
				parent=layer.getBottom(0);
			}
			
			String bottom = layer.getName();
			
			List<LayerParameter> curLayers=getAllBottoms(bottom);
			
			//remove concat layer if any from this and set it to global concat
			Iterator<LayerParameter> it=curLayers.iterator();
			while(it.hasNext()) {
				LayerParameter entry=it.next();
				if (entry.getType().equals("Concat")) {
					concatLayer=entry;
					it.remove();
				}
			}
			
			//add one layer for thyself
			String scope=null;
			if (layer.getName().contains("/")) {
				// TODO check for all type of invalid variable naming in python
				String[] temp=layer.getName().split("/");
				scope=temp[temp.length-1];
			}
			else {
				// if no slash there 
				scope=layer.getName();
			}
			
			if (curLayers.size()==0) {
				//change made for multiplexing. net variable which is a output of selectinput function will go as bottom
				branch_scope_code += createSimpleLayer(layer, branch_name,"net",scope);
			}
			
			if (curLayers.size()==1) {
				//we have a chaining layer forward. 
				//if it's a convolution layer, then code for the first one would change a bit
				if (layer.getType().equals("Convolution")) {
					try {
						branch_scope_code+= createSimpleConvolutionEndPointWithConfiguredOutputs(layer, branch_name, "net", scope, concatLayer.getName());
					}
					catch(Exception e) {
						System.out.println(e);
						System.out.println(layer);
					}
				}
				else {
					branch_scope_code += createSimpleLayer(layer, branch_name,"net",scope);
				}
				
				LayerParameter branch=curLayers.get(0);
				//TODO boilerplate code
				scope=null;
				if (branch.getName().contains("/")) {
					// TODO check for all type of invalid variable naming in python
					String[] temp=branch.getName().split("/");
					scope=temp[temp.length-1];
				}
				else {
					// if no slash there 
					scope=branch.getName();
				}
				
				
				// now check bottoms, and remove concat
				List<LayerParameter> forward=getAllBottoms(branch.getName());
				it=forward.iterator();
				while(it.hasNext()) {
					LayerParameter entry=it.next();
					if (entry.getType().equals("Concat")) {
						concatLayer=entry;
						it.remove();
					}
				}
				
				if (forward.size()==0) {
					//no further chaining
					//just write this one function
					// this could be done in prior step, but just staying safe with inceptionv1
					branch_scope_code += createSimpleLayer(branch,branch_name,null, scope);
				}
				else if(forward.size()==1){
					//chaining
					if (branch.getType().equals("Convolution")) {
						branch_scope_code+= createSimpleConvolutionEndPointWithConfiguredOutputs(branch, branch_name, "net", scope, concatLayer.getName());
					}
					else {
						branch_scope_code += createSimpleLayer(branch, branch_name,"net",scope);
					}
					branch_scope_code+=chaining(forward.get(0),branch_name,null, scope,concatLayer.getName());
				}
				else {
					//create its own
					if (branch.getType().equals("Convolution")) {
						branch_scope_code+= createSimpleConvolutionEndPointWithConfiguredOutputs(branch, branch_name, "net", scope, concatLayer.getName());
					}
					else {
						branch_scope_code += createSimpleLayer(branch, branch_name,"net",scope);
					}
					//create new branch
					//Branching Recursion
					Map<String, String> temp=createNewBranch(forward,"");
					branch_scope_code+=temp.get("code");
					// no need to update bottom
					//bottom=dictionary.get("bottom");
					branch_scope_code+= branch_name + "=" + concatLayerCommonName;
				}
			}
			else if (curLayers.size()>1) {
				//create new branch
				//Branching Recursion
				Map<String, String> temp=createNewBranch(curLayers,"");
				branch_scope_code+=dictionary.get("code");
				// no need to update bottom
				//bottom=dictionary.get("bottom");
				branch_scope_code+= branch_name + "=" + concatLayerCommonName;
			}
			
			mixed_scope_code+=branch_scope_header+tabSpaceAllLines(branch_scope_code)+"\n";
			
		}
		
		//end with concating the layers
		if (concatLayer!=null) {
			String root=concatLayer.getName();
			mixed_scope_header=makeWithHeader("tf.variable_scope", Arrays.asList("'"+root+"'"))+'\n';
			
			//Created concat layer
			mixed_scope_code+= createConcatLayer(concatLayer);
			
			//template code in the beginning of each branching code unique to multiplexing
			code="\n############## template code added for multiplexing ##############\n" + 
					"net = selectinput('"+ root+"', end_points['"+parent+"'])\n" + 
					"############## end template code ##############\n";
			
			code+= mixed_scope_header+ tabSpaceAllLines(mixed_scope_code) + "\n";
			
			//put the full thing in end_points
			code += "end_points['" + root + "']="+concatLayerCommonName+"\n";
			
			dictionary.put("code", code);
			dictionary.put("bottom", root);
		}
		else {
			//last output layer 
			//Limitation: this assumption
			String root="LogitsNasif";
			mixed_scope_header=makeWithHeader("tf.variable_scope", Arrays.asList("'"+root+"'"))+'\n';
			code+= mixed_scope_header+ tabSpaceAllLines(mixed_scope_code) + "\n";
			
			//put the full thing in end_points
			code += "end_points['" + root + "']=LogitsNasif\n";
			
			dictionary.put("code", code);
			dictionary.put("bottom", null);
		}
		
		return dictionary;
	}
	
	protected String chaining(LayerParameter branch, String branch_name, String bottom, String scope, String root) {
		String code="";
		
		//TODO boilerplate code
		scope=null;
		if (branch.getName().contains("/")) {
			// TODO check for all type of invalid variable naming in python
			String[] temp=branch.getName().split("/");
			scope=temp[temp.length-1];
		}
		else {
			// if no slash there 
			scope=branch.getName();
		}
		
		
		// now check bottoms, and remove concat
		List<LayerParameter> forward=getAllBottoms(branch.getName());
		Iterator<LayerParameter> it=forward.iterator();
		while(it.hasNext()) {
			LayerParameter entry=it.next();
			if (entry.getType().equals("Concat")) {
				it.remove();
			}
		}
		
		if (forward.size()==0) {
			//no further chaining
			//just write this one function
			// this could be done in prior step, but just staying safe with inceptionv1
			code += createSimpleLayer(branch,branch_name,null, scope);
		}
		else if(forward.size()==1){
			//chaining
			if (branch.getType().equals("Convolution")) {
				code+= createSimpleConvolutionEndPointWithConfiguredOutputs(branch, branch_name, "net", scope, root);
			}
			else {
				code += createSimpleLayer(branch, branch_name,"net",scope);
			}
			code+=chaining(forward.get(0),branch_name,null, scope,root);
		}
		else {
			//create its own
			if (branch.getType().equals("Convolution")) {
				code+= createSimpleConvolutionEndPointWithConfiguredOutputs(branch, branch_name, "net", scope,root);
			}
			else {
				code += createSimpleLayer(branch, branch_name,"net",scope);
			}
			//create new branch
			//Branching Recursion
			Map<String, String> temp=createNewBranch(forward,"");
			code+=temp.get("code");
			// no need to update bottom
			//bottom=dictionary.get("bottom");
			code+= branch_name + "=" + concatLayerCommonName;
		}
		
		return code;
	}
	
	@Override
	protected String createSimpleConvolutionEndPoint(LayerParameter layer, String name, String bottom, String scope) {
		//check if end_point already exists
		if (end_points.contains(layer.getName())) {
			return "";
		}
		else {
			end_points.add(layer.getName());
		}
		
		String code="";
		
		ConvolutionParameter typeParam=layer.getConvolutionParam();
		
		//left side 
		// name 
		if (name !=null) {
			code+=name;
		}
		else {
			code+="end_points['"+layer.getTop(0)+"']";
		}
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		if (bottom != null) {
			if(bottom.equals("net") || bottom.contains("branch")) {
				//no change made
			}
			else {
				bottom = "end_points['"+bottom+"']";
			}
			
		}
		else {
			bottom = "end_points['"+layer.getBottom(0)+"']";
		}
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
			if(typeParam.getKernelSizeCount()>0 && typeParam.getKernelSize(0)>0) {
				s+=Integer.toString(typeParam.getKernelSize(0));
			}
			else {
				s+="7";
			}
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			if(typeParam.getKernelSizeCount()>0 && typeParam.getKernelSize(0)>0) {
				s+=Integer.toString(typeParam.getKernelSize(0));
			}
			else {
				s+="7";
			}
		}
		s+="]";
		arguments.add(s);
		
		//stride
		//assuming single stride
		if (typeParam.getStrideList().size()>1) {
			errors.add("more than one stride for: "+layer.getName());
		}
		else if(typeParam.getStrideList().size()==1) {
			arguments.add("stride="+Integer.toString(typeParam.getStride(0)));
		}
		else {
			arguments.add("stride=1");
		}
		
		//scope
		if (scope!=null) {
			arguments.add("scope='"+scope+"'");
		}
		else {
			arguments.add("scope='"+layer.getTop(0)+"'");
		}
		
		//make slim call
		code+="slim.conv2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		return code;
	}
	protected String createSimpleConvolutionEndPointWithConfiguredOutputs(LayerParameter layer, String name, String bottom, String scope, String root) {
		//check if end_point already exists
		if (end_points.contains(layer.getName())) {
			return "";
		}
		else {
			end_points.add(layer.getName());
		}
		
		String code="############## code changes for multiplexing ##############\n" + 
				"# The number of filters (argument name: num_outputs) is 96 in the original model . \n" + 
				"# In the multiplexing code, the value can be reconfigured by the config argument.\n";
		
		ConvolutionParameter typeParam=layer.getConvolutionParam();
		
		//left side 
		// name 
		if (name !=null) {
			code+=name;
		}
		else {
			code+="end_points['"+layer.getTop(0)+"']";
		}
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		if (bottom != null) {
			if(bottom.equals("net") || bottom.contains("branch")) {
				//no change made
			}
			else {
				bottom = "end_points['"+bottom+"']";
			}
			
		}
		else {
			bottom = "end_points['"+layer.getBottom(0)+"']";
		}
		arguments.add(bottom);
		
		//add num_of_output
		String num_output=Integer.toString(typeParam.getNumOutput());
		num_output="selectdepth('"+root+"',"+num_output+")";
		arguments.add(num_output);
		
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
			if(typeParam.getKernelSizeCount()>0 && typeParam.getKernelSize(0)>0) {
				s+=Integer.toString(typeParam.getKernelSize(0));
			}
			else {
				s+="7";
			}
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			if(typeParam.getKernelSizeCount()>0 && typeParam.getKernelSize(0)>0) {
				s+=Integer.toString(typeParam.getKernelSize(0));
			}
			else {
				s+="7";
			}
		}
		s+="]";
		arguments.add(s);
		
		//stride
		//assuming single stride
		if (typeParam.getStrideList().size()>1) {
			errors.add("more than one stride for: "+layer.getName());
		}
		else if(typeParam.getStrideList().size()==1) {
			arguments.add("stride="+Integer.toString(typeParam.getStride(0)));
		}
		else {
			arguments.add("stride=1");
		}
		
		//scope
		if (scope!=null) {
			arguments.add("scope='"+scope+"'");
		}
		else {
			arguments.add("scope='"+layer.getTop(0)+"'");
		}
		
		//make slim call
		code+="slim.conv2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		
		code+="############## end code changes##############\n";
		
		removeHandledLayer(layer);
		
		return code;
	}
	@Override
	protected String createSimplePoolingEndPoint(LayerParameter layer, String name, String bottom, String scope) {
		//check if end_point already exists
		if (end_points.contains(layer.getName())) {
			return "";
		}
		else {
			end_points.add(layer.getName());
		}
		
		String code="";
		
		PoolingParameter typeParam=layer.getPoolingParam();
		
		//left side 
		// name 
		if (name !=null) {
			code+=name;
		}
		else {
			code+="end_points['"+layer.getTop(0)+"']";
		}
		
		code+="=";
		
		//right side
		//sequence in arguments is important
		List<String> arguments=new ArrayList<String>();
		 
		//add bottom 
		//assuming bottom would be only one for Convolution Layer
		if (bottom != null) {
			if(bottom.equals("net")) {
				//no change made
			}
			else {
				bottom = "end_points['"+bottom+"']";
			}
			
		}
		else {
			bottom = "end_points['"+layer.getBottom(0)+"']";
		}
		
		arguments.add(bottom);
		
		
		//kernel dimension
		String s="[";
		if (typeParam.hasKernelH()) {
			s+=Integer.toString(typeParam.getKernelH());
		}
		else {
			if (typeParam.getKernelSize()>0) {
				s+=Integer.toString(typeParam.getKernelSize());
			}
			else {
				s+="7";
			}
		}
		s+=",";
		if (typeParam.hasKernelW()) {
			s+=Integer.toString(typeParam.getKernelW());
		}
		else {
			if (typeParam.getKernelSize()>0) {
				s+=Integer.toString(typeParam.getKernelSize());
			}
			else {
				s+="7";
			}
		}
		s+="]";
		arguments.add(s);
		
		//stride
		if(typeParam.hasStride()) {
			arguments.add("stride="+Integer.toString(typeParam.getStride()));
		}
		else {
			arguments.add("stride=1");
		}
		
		//scope
		if (scope!=null) {
			arguments.add("scope='"+scope+"'");
		}
		else {
			arguments.add("scope='"+layer.getTop(0)+"'");
		}
		
		//make slim call
		code+="slim.max_pool2d";
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		return code;
	}

}
