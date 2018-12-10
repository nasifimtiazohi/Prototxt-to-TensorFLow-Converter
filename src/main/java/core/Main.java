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
	
	static String prototxtInput;
	static File simpleOutput;
	static File multiplexingOutput;
	public static void main(String[] args) {
		
		// work on given 3 input files
		String[] myInputFiles=new String[] {"inceptionv1.prototxt","alexnet.prototxt","lenet.prototxt"};
		for (int i=0;i<myInputFiles.length;i++) {
			generateTensorFlowFiles(myInputFiles[i]);
		}
		
		if(args.length > 0) {
			for(int i=0;i<args.length;i++) {
				generateTensorFlowFiles(args[i]);
			}
		}
		
	
	}
	
	private static void generateTensorFlowFiles(String inputFileName) {
		try {
			prototxtInput=FileUtils.readFileToString(new File(inputFileName));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("input file not found");
			e.printStackTrace();
		}
		
		//TODO: identify which type of file? deploy/solver/train_val?
		//TODO: if not deploy type then reject
		
		CodeGenerator codeGen= new CodeGenerator();
		MultiplexingCodeGenerator multGen=new MultiplexingCodeGenerator();
		
		
		//if input is a deploy file containing network Info
		NetParameter.Builder netParameterBuilder=NetParameter.newBuilder();
		try {
			TextFormat.merge(prototxtInput, ExtensionRegistry.getEmptyRegistry(), netParameterBuilder);
		} catch (ParseException e) {
			System.out.println("problem in parsing input prototxt to NetParameter object");
			e.printStackTrace();
		}
		NetParameter netParameter=netParameterBuilder.build();
		
		
		
		// write code to file
		InstantiateOutputFiles(inputFileName);
		FileWriter writer;
		try {
			codeGen.generateNetworkCode(netParameter);
			writer = new FileWriter(simpleOutput);
			writer.write(codeGen.getSimpleTensorFlowPython());
			writer.close();
		} catch (Exception e) {
			System.out.println("problem in writing to output files");
			e.printStackTrace();
		}
		
		try {
			multGen.generateNetworkCode(netParameter);
			writer=new FileWriter(multiplexingOutput);
			writer.write(multGen.getMultiplexingTensorFlowPython());
			writer.close();
		}
		catch (Exception e) {
			System.out.println("problem in writing to output files");
			e.printStackTrace();
		}
		
		//helpful print
		System.out.println("happy ending");

		//System.out.println("Errors for simple tensorflow");
		codeGen.printErrors();
		//System.out.println("Errors for multiplexing tensorflow");
		multGen.printErrors();
	}
	private static void InstantiateOutputFiles(String inputFileName) {
		String filenameWithoutExtension=inputFileName.substring(0, inputFileName.length()-9);
		simpleOutput=new File("simple"+filenameWithoutExtension+".py");
		multiplexingOutput=new File("multiplexing"+filenameWithoutExtension+".py");
	}
	
}
