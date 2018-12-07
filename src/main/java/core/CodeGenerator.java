package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.DropoutParameter;
import caffe.Caffe.InnerProductParameter;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.PoolingParameter;

public class CodeGenerator {
	private String simpleTensorFlowPython;
    private String multiplexingTensorFlowPython;
    private NetParameter net;
    public List<String> errors;
    private List<String> end_points;
    private String input;
    List<LayerParameter> Unhandled_Layers;
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
		Unhandled_Layers=new ArrayList(net.getLayerList()); //to use iterator remove 
		importStatements();
		networkFunction();
		commonCodeToAllModels();
	}

	private void commonCodeToAllModels() {
		String code="\n\n\n"
				+ "# The below code is applicable to any model. It is adapted from \n" + 
				"# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_utils.py\n" + 
				"def default_arg_scope(is_training=True, \n" + 
				"                        weight_decay=0.00004,\n" + 
				"                        use_batch_norm=True,\n" + 
				"                        batch_norm_decay=0.9997,\n" + 
				"                        batch_norm_epsilon=0.001,\n" + 
				"                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):\n" + 
				"\n" + 
				"  batch_norm_params = {\n" + 
				"      # Decay for the moving averages.\n" + 
				"      'decay': batch_norm_decay,\n" + 
				"      # epsilon to prevent 0s in variance.\n" + 
				"      'epsilon': batch_norm_epsilon,\n" + 
				"      # collection containing update_ops.\n" + 
				"      'updates_collections': batch_norm_updates_collections,\n" + 
				"      # use fused batch norm if possible.\n" + 
				"      'fused': None,\n" + 
				"  }\n" + 
				"  if use_batch_norm:\n" + 
				"    normalizer_fn = slim.batch_norm\n" + 
				"    normalizer_params = batch_norm_params\n" + 
				"  else:\n" + 
				"    normalizer_fn = None\n" + 
				"    normalizer_params = {}\n" + 
				"\n" + 
				"  # Set training state \n" + 
				"  with slim.arg_scope([slim.batch_norm, slim.dropout],\n" + 
				"                        is_training=is_training):\n" + 
				"    # Set weight_decay for weights in Conv and FC layers.\n" + 
				"    with slim.arg_scope([slim.conv2d, slim.fully_connected],\n" + 
				"                        weights_regularizer=slim.l2_regularizer(weight_decay)):\n" + 
				"      # Set batch norm \n" + 
				"      with slim.arg_scope(\n" + 
				"          [slim.conv2d],\n" + 
				"          normalizer_fn=normalizer_fn,\n" + 
				"          normalizer_params=normalizer_params):\n" + 
				"          # Set default padding and stride\n" + 
				"            with slim.arg_scope([slim.conv2d, slim.max_pool2d],\n" + 
				"                      stride=1, padding='SAME') as sc:\n" + 
				"              return sc";
		simpleTensorFlowPython+=code;
		
	}

	private void networkFunction() {
		String header=networkFunctionHeader();
		String code=networkFunctionCode();
		code+="return Logit,end_points";
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
		
		//when we handle a layer, we will remove that item from this list
		
		
		//Handle input
		// create the first end_point for input
		if (net.getInputCount()==1) {
			code+="end_points['"+input+"']="+input+"\n";
		}
		//check if any layer is of type input. if yes, remove it as handle
		Iterator<LayerParameter> it= Unhandled_Layers.iterator();
		while(it.hasNext()) {
			LayerParameter entry=it.next();
			if (entry.getType().equals("Input")) {
				it.remove();
			}
		}
		
		//remove all layers of type BatchNorm, Scale, ReLU
		//Limitation: we ignore the above three layers
		it=Unhandled_Layers.iterator();
		while(it.hasNext()) {
			LayerParameter entry=it.next();
			if(entry.getType().equals("BatchNorm") || entry.getType().equals("Scale") || entry.getType().equals("ReLU")) {
				it.remove();
			}
		}
		
		//Note: bottom means input of a layer, top means the own layer
		//Confusion: Assuming top_count will always one
		
		//start with input as bottom
		String bottom=input;
	
		while (Unhandled_Layers.size()>0) {
			
			List<LayerParameter> curLayers=new ArrayList<LayerParameter>();
			
			for(int i=0;i<Unhandled_Layers.size();i++) {
				LayerParameter layer=Unhandled_Layers.get(i);
				List<String> bottomList=layer.getBottomList();
				for (int j=0;j<bottomList.size();j++) {
					if (bottomList.get(j).equals(bottom)) {
						curLayers.add(layer);
						break;
					}
				}
			}
			
			//Limitation: assume input goes to only one layer?
			
			if (curLayers.size()==1) {
				LayerParameter layer=curLayers.get(0);
				if (layer.getType().equals("Concat")) {
					//TODO
				}
				else {
					// simple layer :has only one input that also does not go to any other layer
					code+=createSimpleLayer(layer);
					bottom=layer.getTop(0);
				}
			}
			else if(curLayers.size()>1) {
				
			}
			else {
				System.out.println("debug what happend");
				break;
			}
			
			
		}
		
		
		
		
		
		//OLD CODE
//		List<LayerParameter> layers = net.getLayerList();
//		for (int i=0;i<layers.size();i++) {
//			// search for how many layers take current bottom as top
//			LayerParameter layer=layers.get(i);
//			
//			if (layer.getName().contains("/")==false){
//				code+=createSimpleLayer(layer);
//			}
//			else {
//				code+=createBranchedlayer(layer);
//			}
//		}
//		
//		//TODO check how many
//		for (int i=0;i<layers.size();i++) {
//			LayerParameter layer=layers.get(i);
//			
//			if (layer.getName().contains("/")==false){
//				code+=createSimpleLayer(layer);
//			}
//			else {
//				code+=createBranchedlayer(layer);
//			}
//			
//		}
		
		
		return code;
	}
	
	private void removeHandledLayer(LayerParameter layer) {
		Iterator<LayerParameter> i= Unhandled_Layers.iterator();
		while (i.hasNext()) {
			if (i.next().equals(layer)) {
				i.remove();
			}
		}		
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
		else if(layer.getType().equals("Dropout")) {
			DropoutParameter dorpoutParam=layer.getDropoutParam();
			//left side
			//TODO handle more than one bottom
			String bottom="end_points['"+layer.getBottom(0)+"']";
			code+=bottom;
			
			code+="=";
			
			//right side
			code+="slim.dropout";
			
			List<String> arguments=new ArrayList<String>();
			
			arguments.add(bottom);
			arguments.add(Float.toString(1-dorpoutParam.getDropoutRatio()));
			arguments.add("scope='"+end_point+"'");
			
			code+=makeArgumentList(arguments);
			
			code+="\n";
		}
		else if(layer.getType().equals("Reshape")) {
			//TODO learn squeezing
			//left side
			code+="Logit"; //fixed name for all squeeze op which will be returned eventually
			
			code+="=";
			
			//right side
			code+="tf.squeeze("+parent.toLowerCase()+", [1,2], name='SpatialSqueeze')";
			
			code+="\n";
			code+="end_points['Logit']="+parent.toLowerCase()+"\n";
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
			code+="])\n";
		}
		else if(layer.getType().equals("Softmax")) {
//			//Softmax is probably the last layer
//			//before writing it's own code,
//			//Get the last convolution layer
//			//and squeeze it
//			//TODO learn squeezing
//			LayerParameter lastConvolution=null;
//			for(int i=net.getLayerList().size()-1;i>=0;i--) {
//				if(net.getLayerList().get(i).getType().equals("Convolution")) {
//					lastConvolution=net.getLayerList().get(i);
//					break;
//				}
//			}
//			String lastConvolution_endPoint=null;
//			if (lastConvolution.getName().contains("/")) {
//				String[] temp=lastConvolution.getName().split("/");
//				//TODO remove lowercases from everywhere to remove confusion
//				lastConvolution_endPoint=temp[temp.length-2];
//			}
//			else {
//				lastConvolution_endPoint=lastConvolution.getName();
//			}
//			code+="Logit=tf.squeeze("+lastConvolution_endPoint+", [1,2], name='SpatialSqueeze')\n";
//			code+="end_points['"+lastConvolution_endPoint+"']= Logit\n";
			
			//softmax own code
			//left side
			code+="end_points['"+layer.getName()+"']";
			
			code+="=";
			
			//right side
			code+="slim.softmax";
			
			List<String> arguments=new ArrayList<String>();
			
			//TODO handle more than one bottom error
			arguments.add(layer.getBottom(0));
			arguments.add("scope='"+layer.getTop(0)+"'");
			code+=makeArgumentList(arguments);
			
			code+="\n";
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
		if (typeParam.getStrideList().size()!=1) {
			errors.add("more than one stride for: "+layer.getName());
		}
		arguments.add("stride="+Integer.toString(typeParam.getStride(0)));
		
		//scope
		arguments.add("scope='"+layer.getTop(0)+"'");
		
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
		arguments.add("stride="+Integer.toString(typeParam.getStride()));
		
		//scope
		arguments.add("scope='"+layer.getTop(0)+"'");
		
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
		int output_count=0;
		List<LayerParameter> layers=net.getLayerList();
		for (int i=layers.size()-1;i>=0;i--) {
			LayerParameter layer=layers.get(i);
			if (layer.getType().equals("Convolution") ) {
				ConvolutionParameter convParam = layer.getConvolutionParam();
				output_count=convParam.getNumOutput();
				break;
			}
			else if (layer.getType().equals("InnerProduct")) {
				InnerProductParameter innerProdParam = layer.getInnerProductParam();
				output_count=innerProdParam.getNumOutput();
				break;
			}
		}
		String num_classes="num_classes="+Integer.toString(output_count);
		arguments.add(num_classes);
		
		//Hardcoded assumption
		String train_or_test="is_training=true";
		arguments.add(train_or_test);
		
		//Hardcoded assumption
		String variableReuse="reuse=true";
		arguments.add(variableReuse);
		
		//Scope is the network name itself
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
