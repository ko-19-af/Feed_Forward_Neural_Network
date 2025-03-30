/** This class uses the Feed Forward Neural Network to classify data
 *
 * IDE used: IntelliJ
 *
 * @Author Kevin Olenic
 * ST#: 6814974
 * @Version 1.2
 * @Since 2023-02-15
 */

import java.util.*;
import BasicIO.*;

public class FFNN{
    //private static final ReportPrinter report = new ReportPrinter();
    private static final ReportPrinter report = new ReportPrinter();
    static int[] Topology;// array containing the topology of the FFNN
    static double[] ignitionData;// holds data on engine ignition status
    static double[][] reelData;// holds data on engine reel data
    public static double[][] rTest;// array holding testing data
    public static double[] iTest;// array holding expected output of test data
    public static int run = 1;

    public FFNN(){

        Brain b = new Brain();

        readFile(new ASCIIDataFile());// load first files data
        while(run < 7){
            Topology = new int[]{reelData[0].length, 23, 30, 7, 40, 1};// designate topology of network
            b.initialiseNetwork();// Initialise network weights and biases

            double[][] rTrain = new double[36][48];// array to hold training data
            double[] iTrain = new double[36];// array to hold training expected outputs
            rTest = new double[17][48];// array to hold testing data
            iTest = new double[17];// array to hold testing expected outputs
            for (int y = 0; y < rTrain.length; y++) {// load first training data
                System.arraycopy(reelData[y], 0, rTrain[y], 0, rTrain[0].length);
                iTrain[y] = ignitionData[y];
            }
            for (int y = rTrain.length; y < reelData.length; y++) {// load first training data
                System.arraycopy(reelData[y], 0, rTest[y - rTrain.length], 0, rTest[0].length);
                iTest[y - rTrain.length] = ignitionData[y];
            }

            b.train(rTrain, iTrain, 15000, .5);// train the model
            run++;
        }
        setUpReport();
        print(b.predict(rTest),iTest);
        report.close();// print report
        System.exit(1);// close program
    }

    public static void main(String[] args) {
        new FFNN();
    }// main

    /** This method reads a data file and collects the information on it
     * @param file containing the data being read
     */

    private void readFile(ASCIIDataFile file){
        String[] data = file.readString().split(" ");
        ignitionData = new double[Integer.parseInt(data[0])];
        reelData = new double[Integer.parseInt(data[0])][Integer.parseInt(data[1])];
        for(int y = 0; y < Integer.parseInt(data[0]); y++){// read all data entries
            String[] eData = file.readString().split(" ");// read engine data
            ignitionData[y] = Integer.parseInt(eData[0]);// load ignition data
            int position = 0;// position in reelData
            double normalization = 0;
            for(int x = 1; x < eData.length; x++){//load reel data
               if(!eData[x].equals("")) {// if valid data to enter into array
                   double value = Double.parseDouble(eData[x]);
                   reelData[y][position] = value;// from data read from file
                   normalization += value;
                   position++;// move to next position when data is added
               }
            }
            for(int x = 0; x < reelData[0].length; x++){// normalize the data values
                reelData[y][x] = reelData[y][x]/normalization;
            }
        }
        file.close();// close the file
    }//readFile

    /** Create file format for report2
     *
     */
    private void setUpReport(){

        report.addField("Predicted", "", 25);
        report.addField("Expected", "", 1);

    }//setUpReport

    /** This method prints the predicted values and their associated expected values of a train network
     *
     * @param predicted values from the trained FFNN
     * @param expected values of the training data
     */
    private void print(Hashtable<Integer,double[]> predicted, double[] expected){
        for(int x = 0; x < expected.length; x++) {
            report.writeDouble("Predicted", predicted.get(x)[0]);
            report.writeInt("Expected", (int) expected[x]);
        }
    }//print
}//Main