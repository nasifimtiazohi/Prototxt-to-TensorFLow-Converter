package core;

import java.util.HashMap;

import caffe.Caffe.NetParameter;

public class CodeGenerator {
	private String simpleTensorFlowPython;
    private String multiplexingTensorFlowPython;
    
    public CodeGenerator() {
		setSimpleTensorFlowPython("");
		setMultiplexingTensorFlowPython("");
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
		// TODO Auto-generated method stub
		
	}

	
	
}
