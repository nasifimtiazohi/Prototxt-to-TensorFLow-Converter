package core;

import caffe.Caffe;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;

public class Main {
	static String inputFileName="deploy.prototxt";
	static String prototxtInput;
	static File simpleOutput;
	static File multiplexingOutput;
	public static void main(String[] args) {
		
		//read input prototxt file
		try {
			prototxtInput=FileUtils.readFileToString(new File(inputFileName));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("input file not found");
			e.printStackTrace();
		}
		
		//TODO: identify which type of file? deploy/solver/train_val?
		
		CodeGenerator codeGen= new CodeGenerator();
		
		
		//if input is a deploy file containing network Info
		NetParameter.Builder netParameterBuilder=NetParameter.newBuilder();
		try {
			TextFormat.merge(prototxtInput, ExtensionRegistry.getEmptyRegistry(), netParameterBuilder);
		} catch (ParseException e) {
			System.out.println("problem in parsing input prototxt to NetParameter object");
			e.printStackTrace();
		}
		NetParameter netParameter=netParameterBuilder.build();
		codeGen.generateNetworkCode(netParameter);
		
		// write code to file
		InstantiateOutputFiles();
		FileWriter writer;
		try {
			writer = new FileWriter(simpleOutput);
			writer.write(codeGen.getSimpleTensorFlowPython());
			writer.close();
			writer=new FileWriter(multiplexingOutput);
			writer.write(codeGen.getMultiplexingTensorFlowPython());
			writer.close();
		} catch (IOException e) {
			System.out.println("problem in writing to output files");
			e.printStackTrace();
		}
		
		//helpful print
//		List<LayerParameter> layers=netParameter.getLayerList();
//		for (int i=0;i<layers.size();i++) {
//			System.out.println(layers.get(i).getName());
//		}
		System.out.println("happy ending");
	
	}
	private static void InstantiateOutputFiles() {
		String filenameWithoutExtension=inputFileName.substring(0, inputFileName.length()-9);
		simpleOutput=new File("simple"+filenameWithoutExtension+".py");
		multiplexingOutput=new File("multiplexing"+filenameWithoutExtension+".py");
	}
	
}
