package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import caffe.Caffe.NetParameter;

public class CodeGenerator {
	private String simpleTensorFlowPython;
    private String multiplexingTensorFlowPython;
    private NetParameter net;
    List<String> errors;
    public CodeGenerator() {
		setSimpleTensorFlowPython("");
		setMultiplexingTensorFlowPython("");
		errors=new ArrayList<String>();
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
		
		List<String> arguments=new ArrayList<String>();
		arguments.add("scope");
		arguments.add(net.getName());
		String mainWithHeader=makeWithHeader("tf.variable_scope",arguments,"tf");
		String mainWithBlock="";
		
		mainWithBlock+="end_points_collection = sc.original_name_scope + '_end_points'";
		
		mainWithBlock= tabSpaceAllLines(mainWithBlock);
		code += mainWithHeader + "\n" + mainWithBlock + "\n";
		
		return code;
	}

	private String makeWithHeader(String name, List<String> arguments, String alias) {
		String code="with "+name+"(";
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
		code+=" as "+alias+":";
		return code;
	}

	private String tabSpaceAllLines(String code) {
		System.out.println("entering: "+code);
		
		code="\t"+code;
		for (int i = -1; (i = code.indexOf("\n", i + 1)) != -1; i++) {
			if (i+1 < code.length()) {
			    String before=code.substring(0, i+1);
			    String after=code.substring(i+1);
			    code=before+'\t'+after;
			}
		} 
		
		System.out.println("exiting: "+code);
		
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
		
		String scope="scope="+name;
		arguments.add(scope);
		
		String code=makeFunctionHeader(name,arguments);
		
		return code;
	}

	private String makeFunctionHeader(String name, List<String> arguments) {
		String code="def "+name+"(";
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
		code+=":";
		return code;
	}

	private void importStatements() {
		String code="import tensorflow as tf"
				+ "\n"
//				+ "slim=tf.contrib.slim"
//				+ "\n\n"
				;
		simpleTensorFlowPython += code;
		
	}

	
	
}
