package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import caffe.Caffe.BlobShape;
import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.DropoutParameter;
import caffe.Caffe.InnerProductParameter;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.PoolingParameter;

public class CodeGenerator {
	protected String simpleTensorFlowPython;
    protected NetParameter net;
    protected List<String> errors;
    protected List<String> end_points;
    protected String input;
    protected String concatLayerCommonName="concat_point";
    protected LayerParameter outputLayer; //last convolution or innerproduct layer
    protected List<LayerParameter> Unhandled_Layers;

    public CodeGenerator() {
		setSimpleTensorFlowPython("");
		errors=new ArrayList<String>();
		end_points=new ArrayList<String>();
	}
    
    public void printErrors() {
    	for(int i=0;i<errors.size();i++) {
			System.out.println(errors.get(i));
		}
    }
	public String getSimpleTensorFlowPython() {
		return simpleTensorFlowPython;
	}

	protected void setSimpleTensorFlowPython(String simpleTensorFlowPython) {
		this.simpleTensorFlowPython = simpleTensorFlowPython;
	}

	

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
		
		simpleTensorFlowPython+=importStatements();
		simpleTensorFlowPython+=networkFunction();
		simpleTensorFlowPython+=changeImageSize();
		simpleTensorFlowPython+=commonCodeToAllModels();
	}

	protected String changeImageSize() {
		String code ="\n \n ### change the default image_size based on the input image size specified in prototxt ### \n";
		Long dim=(long) 0;
		try {
			List<BlobShape> blobs=net.getInputShapeList();
			BlobShape blob=blobs.get(blobs.size()-1);
			List<Long> dims=blob.getDimList();
			dim=dims.get(blob.getDimCount()-1);
		}
		catch (Exception e){
			System.out.println(e);
		}

		
		code+= net.getName()+".default_image_size = "+Long.toString(dim);
		return code;
		
	}

	protected String commonCodeToAllModels() {
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
		return code;
		
	}

	protected String networkFunction() {
		String fullCode="";
		
		String header=networkFunctionHeader();
		String code=networkFunctionCode();
		code+="return LogitsNasif,end_points";
		code=tabSpaceAllLines(code);
		fullCode += header + "\n" +code + "\n";
		
		return fullCode;
	}

	protected String networkFunctionCode() {
		String code="";
		
		String variable_scope_header="with tf.variable_scope(scope,\"Model\",reuse=reuse):";
		String variable_scope_code=variableScope();
		variable_scope_code=tabSpaceAllLines(variable_scope_code);
		code+=variable_scope_header+"\n"+variable_scope_code+"\n";
		
		return code;
	}

	protected String variableScope() {
		String code = "";
		
		String arg_scope_header="with slim.arg_scope(default_arg_scope(is_training)):";
		String arg_scope_code=argumentScope();
		arg_scope_code=tabSpaceAllLines(arg_scope_code);
		code+=arg_scope_header+"\n"+arg_scope_code+"\n";
		
		return code;
	}

	protected String argumentScope() {
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
			
			if(bottom==null) {
				break;
			}
			
			List<LayerParameter> curLayers=getAllBottoms(bottom);
			
			//Limitation: assume input goes to only one layer?
			
			if (curLayers.size()==1) {
				LayerParameter layer=curLayers.get(0);
				
				// simple layer :has only one input that also does not go to any other layer
				code+=createSimpleLayer(layer,null,null,null);
				//update bottom
				bottom=layer.getTop(0);
				
			}
			else if(curLayers.size()>1) {
				//Branching
				Map<String, String> dictionary=createNewBranch(curLayers,"");
				code+=dictionary.get("code");
				//update bottom
				bottom=dictionary.get("bottom");
			}
			else {
				System.out.println("debug what happend");
				System.out.println(bottom);
				break;
			}
			
			
		}
		
		System.out.println(bottom);
		System.out.println(Unhandled_Layers);
		
		
		return code;
	}
	
	protected List<LayerParameter> getAllBottoms(String bottom) {
		
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
		
		return curLayers;
	}
	

	protected Map<String, String> createNewBranch( List<LayerParameter> layers, String code) {
		
		Map<String, String> dictionary = new HashMap<String, String>();
		
		String mixed_scope_header="";
		String mixed_scope_code="";
		
		LayerParameter concatLayer=null;
		
		//check all layers that are branched out
		for (int i=0;i<layers.size();i++) {
			String branch_name= "branch_"+Integer.toString(i);
			String branch_scope_header=makeWithHeader("tf.variable_scope", Arrays.asList("'"+branch_name+"'"))+"\n";
			String branch_scope_code="";
			
			LayerParameter layer=layers.get(i);
			
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
			branch_scope_code += createSimpleLayer(layer, branch_name,null,scope);
			
			if (curLayers.size()==1) {
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
					branch_scope_code += createSimpleLayer(branch,branch_name,branch_name, scope);
				}
				else if(forward.size()==1){
					//chaining
					branch_scope_code += createSimpleLayer(branch,branch_name,branch_name, scope);
					branch_scope_code+=chaining(forward.get(0),branch_name,null);
				}
				else {
					//create its own
					branch_scope_code += createSimpleLayer(branch,branch_name,branch_name, scope);
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

	protected String chaining(LayerParameter branch, String branch_name, String bottom) {
		String code="";
		
		//TODO boilerplate code
		String scope=null;
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
			code += createSimpleLayer(branch,branch_name,branch_name, scope);
		}
		else if(forward.size()==1){
			//chaining
			code += createSimpleLayer(branch,branch_name,branch_name, scope);
			code+=chaining(forward.get(0),branch_name,null);
		}
		else {
			//create its own
			code += createSimpleLayer(branch,branch_name,branch_name, scope);
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

	protected LayerParameter findLayerFromName(String top) {
		for(int i=0;i<Unhandled_Layers.size();i++) {
			if(Unhandled_Layers.get(i).getName().equals(top)) {
				return Unhandled_Layers.get(i);
			}
		}
		return null;
	}

	protected String createConcatLayer(LayerParameter layer) {
		String code="";
		
		//left side 
		code+=concatLayerCommonName;
		
		code+="=";
		
		//right side
		code+= "tf.concat";
		
		List<String> arguments= new ArrayList<String>();
		
		//axis
		if (layer.getConcatParam().hasAxis()) {
			arguments.add("axis="+Integer.toString(layer.getConcatParam().getAxis()));
		}
		else {
			arguments.add("axis=3");
		}
		
		//values
		String temp="values=[";
		for (int i=0;i<layer.getBottomCount();i++) {
			temp+= "branch_"+Integer.toString(i);
			if (i<layer.getBottomCount()-1) {
				temp+=",";
			}
		}
		temp+="]";
		arguments.add(temp);
		
		code += makeArgumentList(arguments);
		
		removeHandledLayer(layer);
		
		return code;
	}

	protected void removeHandledLayer(LayerParameter layer) {
		Iterator<LayerParameter> i= Unhandled_Layers.iterator();
		while (i.hasNext()) {
			LayerParameter entry=i.next();
			if (entry.equals(layer)) {
				i.remove();
			}
		}		
	}

	





	protected String createSimpleLayer(LayerParameter layer, String name,  String bottom, String scope ) {
		String code="";
		//check if last_layer
		if(layer.equals(outputLayer)) {
			//assume convolution layer
			
			ConvolutionParameter typeParam = outputLayer.getConvolutionParam();
			
			//handle name
			if(name==null) {
				name=layer.getTop(0);
			}
			
			//handle scope
			if(scope==null) {
				scope=layer.getTop(0);
			}
			
			//handle kernel size
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
			
			//handle stride
			String stride="";
			if (typeParam.getStrideList().size()>1) {
				errors.add("more than one stride for: "+layer.getName());
			}
			else if(typeParam.getStrideList().size()==1) {
				stride="stride="+Integer.toString(typeParam.getStride(0));
			}
			else {
				stride="stride=1";
			}
			
			//handle bottom and generate code
			if (bottom!=null) {
				code+= name + "=slim.conv2d(end_points['"+bottom+"'],num_classes,"+s+","+stride+",activation_fn=None,\n" + 
						"                             normalizer_fn=None,scope='"+scope+"')\n";
			}
			else {
				code+= name + "=slim.conv2d(end_points['"+layer.getBottom(0)+"'],num_classes,"+s+","+stride+",activation_fn=None,\n" + 
						"                             normalizer_fn=None,scope='"+scope+"')\n";
			}
			
			//rest of the logit code
			if (name==null) {
				code+="LogitsNasif="+layer.getName()+"\n";
			}
			else {
				code+="LogitsNasif="+name+"\n";
			}
			
			//squeeze
			//TODO change hardcoding
			code+="LogitsNasif=tf.squeeze(LogitsNasif, [1, 2], name='SpatialSqueeze')"+"\n";
		}
		
		else if (layer.getType().equals("Convolution")) {
			code+=createSimpleConvolutionEndPoint(layer,name,bottom,scope);
		}
		else if (layer.getType().equals("Pooling")) {
			code+=createSimplePoolingEndPoint(layer,name,bottom,scope);
		}
		else if(layer.getType().equals("Concat")) {
			code+=createConcatLayer(layer);
		}
		else if(layer.getType().equals("Dropout")) {
			code+= createDropoutLayer(layer);
		}
		else if(layer.getType().equals("Reshape")) {
			//probably not related to squeezing
			code+="harmless_variable=0\n";
		}
		else if(layer.getType().equals("Softmax")) {
			code+="end_points['"+layer.getTop(0)+"']";
			
			code+="=";
			
			//right side
			code+="slim.softmax";
			
			List<String> arguments=new ArrayList<String>();
			
			//TODO handle more than one bottom error
			arguments.add("LogitsNasif");
			arguments.add("scope='"+layer.getTop(0)+"'");
			code+=makeArgumentList(arguments);
			
			code+="\n";
		}
		removeHandledLayer(layer);
		
		
		
		return code;
	}


	
	protected String createDropoutLayer(LayerParameter layer) {
		String code="";
				
		DropoutParameter dropoutParam=layer.getDropoutParam();
		//left side
		//TODO handle more than one bottom
		code+="end_points['"+layer.getTop(0)+"']";
		
		
		code+="=";
		
		//right side
		code+="slim.dropout";
		
		List<String> arguments=new ArrayList<String>();
		
		//Limitation: assume only one bottom
		arguments.add("end_points['"+layer.getBottom(0)+"']");
		
		//dropout ratio
		if (dropoutParam.hasDropoutRatio()) {
			arguments.add(Float.toString(1-dropoutParam.getDropoutRatio()));
		}
		else {
			//default value
			arguments.add(Float.toString((float) 0.5));
		}
		arguments.add("scope='"+layer.getName()+"'");
		
		code+=makeArgumentList(arguments);
		
		code+="\n";
		
		return code;
	}

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
			if(bottom.contains("branch")) {
				//do nothing
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
		if (typeParam.hasNumOutput()) {
			arguments.add(Integer.toString(typeParam.getNumOutput()));
		}
		else {
			arguments.add(Integer.toString(0));
		}
		
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

	
	
	protected String makeArgumentList(List<String> arguments) {
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
			if(bottom.contains("branch")) {
				//nothing
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
	protected String makeWithHeader(String name, List<String> arguments, String alias) {
		String code="with "+name;
		code+=makeArgumentList(arguments);
		code+=" as "+alias+":";
		return code;
	}

	protected String makeWithHeader(String name, List<String> arguments) {
		String code="with "+name;
		code+=makeArgumentList(arguments);
		code+=":";
		return code;
	}
	protected String tabSpaceAllLines(String code) {
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
		
		String code=makeFunctionHeader(name,arguments);
		
		return code;
	}

	protected String makeFunctionHeader(String name, List<String> arguments) {
		String code="def "+name;
		code+=makeArgumentList(arguments);
		code+=":";
		return code;
	}

	protected String importStatements() {
		String code="import tensorflow as tf"
				+ "\n"
				+ "slim=tf.contrib.slim" // uses slim api 
				+ "\n\n"
				;
		return code;
		
	}

	
	
}
