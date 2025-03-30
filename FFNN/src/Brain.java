import java.io.IOException;
import java.util.*;
import java.io.File;  // Import the File class
import java.io.FileWriter;   // Import the FileWriter class
/** This class contains the training code and methods for the Feed Forward Neural Network
 *
 * IDE used: IntelliJ
 *
 * @Author Kevin Olenic
 * ST#: 6814974
 * @Version 1.2
 * @Since 2023-02-15
 */

public class Brain {

    private Random r = new Random();// for generating random numbers
    private Hashtable<Integer, double[]> post = new Hashtable<>();// store post-activation values of the forward pass
    private Hashtable<Integer, double[]> error = new Hashtable<>();// store the error values for the neurons
    private Hashtable<Integer, double[][]> weightsTable = new Hashtable<>();// table holding weight matrices
    private  Hashtable<Integer, double[]> biasTable = new Hashtable<>();// table holding bias values
    private double momentum = 0.5; // momentum value

    public Brain(){}//constructor

    /**This method creates the hashtable containing the initial weight matrices
     * and biases for each layer
     *
     */
    public void initialiseNetwork(){
        for(int z = 1; z < FFNN.Topology.length; z++){
            // create array to hold weights (y-Neuron, x-weight)
            double[][] weights = new double[FFNN.Topology[z]][FFNN.Topology[z-1]];

            for(int y = 0; y < weights.length; y++){
                for(int x = 0; x < weights[0].length; x++){
                    weights[y][x] = (-1) + r.nextDouble() * (1-(-1));// set weight to value of [-1, 1]
                }
            }

            weightsTable.put(z, weights);// store weight matrix for layer
            biasTable.put(z, new double[FFNN.Topology[z]]);// store bias values for neurons
        }
    }// initialiseNetwork

    /** This method takes the input data and uses it to train the Feed Forward Neutral Network
     *
     * @param data is the 2-D array containing the reel training data (input data)
     * @param expectedV is the array containing the ignition training data (expected outputs)
     * @param epochs is the number of training steps
     * @param learningRate is the initial learning rate being used to train
     */
    public void train(double[][] data, double[] expectedV, int epochs, double learningRate){
        int random;// initialize variable for holding value of randomly chosen data
        double[] tGE = new double[epochs];
        double[] trGE = new double[epochs];
        try {
            File myObj = new File("C:\\Users\\keole\\Downloads\\4P80 Assign 2 Data\\error"+FFNN.run+".txt");
            if (myObj.createNewFile()) {
                System.out.println("File created: " + myObj.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        for(int e = 1; e <= epochs; e++){// for number of training steps
            double trainGE = 0;// reset the global error for each epoch
            double testGE = 0;
            Stack<Integer> stack = new Stack<>();// stack for holding already processed
            Hashtable<Integer, double[][]> oldWChange = new Hashtable<>();// store old values

            for (double ignored : expectedV) {// for training data size

                // randomly select a training data
                if (stack.empty()) random = (int) (Math.random() * expectedV.length);// if on first data set
                else {
                    do {
                        random = (int) (Math.random() * expectedV.length);//randomly choose training data set
                    } while (stack.contains(random));// while trying to use data already used in training for epoch
                }stack.add(random);// add used training data to stack

                if(Math.abs(expectedV[random] - forwardPass(data[random])[0]) >= .3) learningRate = .8;
                else if(Math.abs(expectedV[random] - forwardPass(data[random])[0]) < .2 &&
                        Math.abs(expectedV[random] - forwardPass(data[random])[0]) >= .05) learningRate = .25;
                else learningRate = 0.001;

                gradient(data[random], expectedV[random]);// calculate the error at each neuron

                for (int t = FFNN.Topology.length-1; t > 0; t--) {// for each layer with weight matrix
                    double[][] change = new double[weightsTable.get(t).length][weightsTable.get(t)[0].length];

                    for (int z = 0; z < weightsTable.get(t)[0].length; z++) {// for each neuron
                        for (int y = 0; y < weightsTable.get(t).length; y++) {// for each connection

                            change[y][z] = learningRate * post.get(t-1)[z] * error.get(t)[y];// calculate adjustment
                            weightsTable.get(t)[y][z] += change[y][z];// change weight

                            // add momentum from old weight (PART C)
                            if(trainGE != 0) weightsTable.get(t)[y][z] +=  momentum * oldWChange.get(t)[y][z];
                        }
                    }
                    oldWChange.put(t, change);// store change matrix for use in momentum
                    for (int y = 0; y < weightsTable.get(t).length; y++) {// update bias values for layer
                        biasTable.get(t)[y] += learningRate * error.get(t)[y];
                    }
                }
                trainGE += Math.abs(expectedV[random] - post.get(FFNN.Topology.length-1)[0]);// get error for epoch
            }

            Hashtable<Integer, double[]> testP = predict(FFNN.rTest);// get predicted values for test data for epoch
            for(int x = 0; x < FFNN.iTest.length; x++){// calculate the global error for the test data
                testGE += Math.abs(testP.get(x)[0] - FFNN.iTest[x]);
            }

            tGE[e-1] = testGE;// store global error for test set for epoch
            trGE[e-1] = trainGE;// store global error for training set for epoch
            if(testGE < 4) return;
        }

        try {
            FileWriter myWriter = new FileWriter("C:\\Users\\keole\\Downloads\\4P80 Assign 2 Data\\error"+FFNN.run+".txt");
            //for(int x = 0; x < epochs;x++) myWriter.write(trGE[x] + "\t" + tGE[x] + "\n");
            for(int x = 0; x < epochs;x++) myWriter.write(tGE[x] + "\n");
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException er) {
            System.out.println("An error occurred.");
            er.printStackTrace();
        }
    }// train

    /** This method performs the training forward pass and uses that information to calculate the delta value
     * component of the weight change formula
     *
     * @param data is the input data going into the network
     * @param output is the expected output from the training data
     */

    private void gradient(double[] data, double output){
        forwardPass(data);// perform forward pass with training data

        for(int l = FFNN.Topology.length-1; l > 0; l--){// calculate error for hidden layer

            double[] errorV = new double[weightsTable.get(l).length];// initialise array to hold error values
            double[] postV = post.get(l);// get post activation values of layer

            for(int x = 0; x < postV.length; x++){// calculate error for each neuron
                // calculate error at output layer
                if(l == FFNN.Topology.length-1) errorV[x] = activativationDeriv(postV[x]) * (output - postV[x]);

                else {// calculate error at hidden layer
                    for (int y = 0; y < post.get(l+1).length; y++) {// for each weight
                        errorV[x] += activativationDeriv(postV[x]) * weightsTable.get(l+1)[y][x] * error.get(l+1)[y];
                    }
                }
            }
            error.put(l, errorV);// store backpropagation weight change values
        }
    }//gradient

    /**This method performs a forward pass on some provided input data and returns
     * the predicted output
     *
     * @param data being pass forward through the neural network
     * @return the post activation values of the output layer (predicted output)
     */
    private double[] forwardPass(double[] data){
        post.put(0, data);// load input data as first pre-activation values
        for(int x = 0; x < FFNN.Topology.length-1; x++){// for each weight matrix
            // calculate pre-activation values of layer then calculate post and then store them
            post.put(x+1, activation(multiply(post.get(x), weightsTable.get(x+1), biasTable.get(x+1))));
        }
        return post.get(FFNN.Topology.length-1);// return post activation values for the output layer
    }//forwardPass

    /** This method multiples the post-activation values of a layer against the weights then adds the bias
     * to calculate the pre activation values for a layer
     *
     * @param in are the post-activation values of the previous layer
     * @param weights are the weights of the layer being multiplied against the post-activation values
     * @param bias is the values being added to the weights
     * @return the pre-activation values of the layer
     */
    private static double[] multiply(double[] in, double[][] weights, double[] bias){
        double[] result = new double[weights.length];// initialise pre-activation values array
        for(int y = 0; y < weights.length; y++){// for each neuron
            for(int x = 0; x < weights[0].length; x++){// for each connection (weight)
                result[y] += in[x] * (weights[y][x]);// input going into neuron times its weight
             }
            result[y] += bias[y];// add the bias of the neuron to pre-activation value
        }
        return result;// return pre-activation values for layer
    }//multiply

    /** This method applies the sigmoid function to calculate the post activation
     * values for a layer
     *
     * @param preActivation values from the hidden layer
     * @return the post activation values
     */
    private double[] activation(double[] preActivation){
        double[] postActivation = new double[preActivation.length];// array to store post activation values
        for(int x = 0; x < preActivation.length; x++){// for each pre-activation value
            //postActivation[x] = Math.tanh(preActivation[x]);// tanh activation function
            postActivation[x] = 1.0/(1.0 + Math.exp(-preActivation[x]));// pass values through sigmoid function
        }
        return postActivation;// return post activation values
    }//activation

    /** This method
     *
     * @param x value being passed through the sigmoid derivative
     * @return the value generated by the sigmoid derivative
     */
    private static double activativationDeriv(double x){
        //return 1 - Math.pow(Math.tanh(x),2);// tanh derivative
        return x * (1 - x);// sigmoid derivative
    }//activationDeriv

    /**This method takes the input data and feeds it into the neural network to make a prediction on the
     * expected outcome of the data
     *
     * @param input is the data being tested
     * @return the hashtable containing the predicted values of the data
     */
    public Hashtable<Integer, double[]> predict(double[][] input){
        Hashtable<Integer, double[]> predictions = new Hashtable<>();// table to hold predictions
        for(int x = 0; x < input.length; x++){// for each input
            predictions.put(x,forwardPass(input[x]));//perform forward pass on data and store predicted output
        }
        return predictions;// return the hashtable containing the predictions
    }// predict
}//Train
