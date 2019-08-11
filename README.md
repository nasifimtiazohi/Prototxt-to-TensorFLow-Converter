# Wootz
Compiler to convert Prototxt to TensorFlow python

## Structure of the Code
src/main/java contains 2 packages - 1) core (written by me) 2) caffe (contains Caffe.java created from protoc)

core package contains 3 files -
1) Main.java - takes input file, instantiate code generator classes, and write the output file
2) CodeGenerator.java - class to generate simple TensorFlow
3) MultiplexingCodeGenerator.java - extends CodeGenerator.java, generates multiplexing code

## Test Cases -
There are three test cases in the file-
1)inceptionv1.prototxt
2)alexnet.prototxt
3)lenet.prototxt

Upon executing Wootz, the program will automatically generate output for these three test files. (Hardcoded)

## How to EXECUTE

The file contains Wootz.jar, a runnable jar file of the project.

Execute the jar files with the name of the input files as program arguments. 
The program can take any number of input files at one go. 

<code> java -jar Wootz.jar filename1.prototxt filename2.prototxt ... </code>

## Understanding Outputs

For each input file and test cases,
two output file will be generated 
as simple and multiplexing TensorFlow code.

example: input - alexnet.prototxt
		 output - simplealexnet.py , multiplexingalexnet.py
		 
## Known Limitations - 

1) For branching of the layers, there must be a concat layer at the end (except for the output layer).
   The code fails if there is branching, but the branched layers do not merge in the concat layer in the end.
   Only during output layer, there can be some branching without any concat layer.

2) Wootz only handles 10 types of layers as stated in the report.

3) if the input is in any other format than the way it is specified in the test files,
	Wootz may fail to read and write image size as 0 (default).
	




