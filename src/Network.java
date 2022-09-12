import java.util.*;
import java.io.*;

/*
 * Defines an A-B-C-D neural network model with two hidden layers. Utilizes backpropagation to
 * train the weights. Obtains the appropriate values for the network variables from an external
 * configuration file. Contains the following methods:
 * main(String[] args)
 * config()
 * allocateRun()
 * allocateTrain()
 * loadTable(String file)
 * activation(double theta)
 * activationPrime(double theta)
 * runEntryRun(int entry)
 * runEntryTrain(int entry)
 * run()
 * reportRun(int entry)
 * error(int entry)
 * randomNumber(double lowerLimit, double upperLimit)
 * randomWeights(double lowerLimit, double upperLimit)
 * deltaWeights()
 * train()
 * reportTrain()
 * printWeights()
 * writeWeights()
 * @author Rohan Rashingkar
 * @version 5/8/22
 */

public class Network
{
   public static int numHiddenLayers;
   public static int numActivationNodes;
   public static int numHiddenNodesFirst;
   public static int numHiddenNodesSecond;
   public static int numOutputNodes;
   public static int numTruthEntries;
   public static double time;

   public static boolean training;
   public static boolean preloaded;
   public static boolean interval;
   public static boolean printWeights;
   public static int intervalValue;
   public static String inputWeights;
   public static String inputTruth;
   public static String outputWeights;
   public static String userFile;

   public static double[] activationNodes;
   public static double[] hiddenNodesFirst;
   public static double[] hiddenNodesSecond;
   public static double[] outputNodes;
   public static double[][] weightsMK;
   public static double[][] weightsKJ;
   public static double[][] weightsJI;
   public static double[][] truthInputs;
   public static double[][] truthOutputs;

   public static int numIterations;
   public static int maxIterations;
   public static double lower;
   public static double upper;
   public static double lambda;
   public static double error;
   public static double threshold;
   public static double[] thetaJ;
   public static double[] thetaK;
   public static double[] psiI;
   public static double[] psiJ;

   /*
    * Creates a network and executes it according to the configuration file
    */
   public static void main(String[] args) throws IOException
   {
      // Reading terminal input
      userFile = "C:\\Users\\scare\\IdeaProjects\\NeuralNetwork\\";
      if (args.length > 0)
      {
         userFile += args[0];
      }
      else
      {
         userFile += "config";
      }
      runConfig();
   } // public static void main(String[] args) throws IOException

   /*
    * Loads the appropriate values based on the configuration file
    */
   public static void runConfig() throws IOException
   {
      // Allocating the appropriate arrays
      config();
      if (training)
      {
         allocateTrain();
      }
      else
      {
         allocateRun();
      }
      loadTable(inputTruth);

      // Assigning initial weights
      if (preloaded)
      {
         Scanner scanner = new Scanner(new FileReader(inputWeights));

         scanner.nextLine();
         scanner.nextLine();
         scanner.nextLine();

         // mk weights
         for (int m = 0; m < numActivationNodes; m++)
         {
            for (int k = 0; k < numHiddenNodesFirst; k++)
            {
               weightsMK[m][k] = Double.parseDouble(scanner.nextLine().split(" ")[2]);
            }
         }
         scanner.nextLine();
         scanner.nextLine();

         // kj weights
         for (int k = 0; k < numHiddenNodesFirst; k++)
         {
            for (int j = 0; j < numHiddenNodesSecond; j++)
            {
               weightsKJ[k][j] = Double.parseDouble(scanner.nextLine().split(" ")[2]);
            }
         }
         scanner.nextLine();
         scanner.nextLine();

         // ji weights
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            for (int i = 0; i < numOutputNodes; i++)
            {
               weightsJI[j][i] = Double.parseDouble(scanner.nextLine().split(" ")[2]);
            }
         }
         scanner.close();
      }
      else
      {
         randomWeights(lower, upper);
      }

      // Executing the appropriate mode
      if (training)
      {
         train();
      }
      else
      {
         run();
      }
   } // public static void runConfig() throws IOException

   /*
    * Assigns values to the instance variables of the network based on the configuration file
    */
   public static void config() throws FileNotFoundException
   {
      String filePath = "C:\\Users\\scare\\IdeaProjects\\NeuralNetwork\\";
      Scanner scanner = new Scanner(new FileReader(userFile));

      // Reading the mode
      training = Boolean.parseBoolean(scanner.nextLine().split(" ")[0]);


      // Reading the number of hidden layers
      numHiddenLayers = Integer.parseInt(scanner.nextLine().split(" ")[0]);

      // Reading the network type
      String[] type = scanner.nextLine().split(" ");
      numActivationNodes = Integer.parseInt(type[0]);
      numHiddenNodesFirst = Integer.parseInt(type[1]);
      numHiddenNodesSecond = Integer.parseInt(type[2]);
      numOutputNodes = Integer.parseInt(type[3]);

      // Reading the number of test cases
      numTruthEntries = Integer.parseInt(scanner.nextLine().split(" ")[0]);

      // Reading the error threshold and maximum iterations
      String[] boundaries = scanner.nextLine().split(" ");
      threshold = Double.parseDouble(boundaries[0]);
      maxIterations = Integer.parseInt(boundaries[1]);

      // Reading the lambda
      lambda = Double.parseDouble(scanner.nextLine().split(" ")[0]);

      // Reading the random weight lower and upper limits
      String[] limits = scanner.nextLine().split(" ");
      lower = Double.parseDouble(limits[0]);
      upper = Double.parseDouble(limits[1]);

      // Reading the boolean for whether the weights are preloaded
      preloaded = Boolean.parseBoolean(scanner.nextLine().split(" ")[0]);

      // Reading the boolean for whether the weights are saved in intervals
      interval = Boolean.parseBoolean(scanner.nextLine().split(" ")[0]);

      // Reading the interval at which weights are saved
      intervalValue = Integer.parseInt(scanner.nextLine().split(" ")[0]);

      // Reading the name of the file from which starting weights are inputted
      inputWeights = filePath + scanner.nextLine().split(" ")[0];

      // Reading the name of the file from which the truth table is inputted
      inputTruth = filePath + scanner.nextLine().split(" ")[0];

      // Reading the name of the file to which trained weights will be outputted
      outputWeights = filePath + scanner.nextLine().split(" ")[0];

      // Reading the boolean for whether weights are printed
      printWeights = Boolean.parseBoolean(scanner.nextLine().split(" ")[0]);

      scanner.close();
   } // public static void config() throws FileNotFoundException

   /*
    * Instantiates the array instance variables that will be used in the run method and assigns
    * them the appropriate amount of memory
    */
   public static void allocateRun()
   {
      activationNodes = new double[numActivationNodes];
      hiddenNodesFirst = new double[numHiddenNodesFirst];
      hiddenNodesSecond = new double[numHiddenNodesSecond];
      outputNodes = new double[numOutputNodes];

      truthInputs = new double[numTruthEntries][numActivationNodes];
      truthOutputs = new double[numTruthEntries][numOutputNodes];

      weightsMK = new double[numActivationNodes][numHiddenNodesFirst];
      weightsKJ = new double[numHiddenNodesFirst][numHiddenNodesSecond];
      weightsJI = new double[numHiddenNodesSecond][numOutputNodes];
   } // public static void allocateRun()

   /*
    * Instantiates the array instance variables that will be used in the train method and assigns
    * them the appropriate amount of memory
    */
   public static void allocateTrain()
   {
      activationNodes = new double[numActivationNodes];
      hiddenNodesFirst = new double[numHiddenNodesFirst];
      hiddenNodesSecond = new double[numHiddenNodesSecond];
      outputNodes = new double[numOutputNodes];

      truthInputs = new double[numTruthEntries][numActivationNodes];
      truthOutputs = new double[numTruthEntries][numOutputNodes];

      weightsMK = new double[numActivationNodes][numHiddenNodesFirst];
      weightsKJ = new double[numHiddenNodesFirst][numHiddenNodesSecond];
      weightsJI = new double[numHiddenNodesSecond][numOutputNodes];

      thetaJ = new double[numHiddenNodesSecond];
      thetaK = new double[numHiddenNodesFirst];
      psiI = new double[numOutputNodes];
      psiJ = new double[numHiddenNodesSecond];
   } // public static void allocateTrain()

   /*
    * Fills the truth table arrays based on the inputted truth table file
    * @param file the inputted truth table
    */
   public static void loadTable(String file) throws IOException
   {
      Scanner scanner = new Scanner(new File(file));
      for (int entry = 0; entry < numTruthEntries; entry++)
      {
         for (int m = 0; m < numActivationNodes; m++)
         {
            truthInputs[entry][m] = scanner.nextDouble();
         }
         for (int i = 0; i < numOutputNodes; i++)
         {
            truthOutputs[entry][i] = scanner.nextDouble();
         }
      }
   } // public static void loadTable(String file) throws IOException

   /*
    * Calculates the activation function for the theta input.
    * In this case, the activation is a sigmoid function
    * @param theta the input to the function
    * @return the value of the activation function for theta
    */
   public static double activation(double theta)
   {
      return 1.0 / (1.0 + Math.exp(-theta));
   }

   /*
    * Calculates the derivative of the activation function for the
    * theta input
    * @param theta the input to the function
    * @return the derivative of the activation function at theta
    */
   public static double activationPrime(double theta)
   {
      double activationValue = activation(theta);
      return activationValue * (1.0 - activationValue);
   }

   /*
    * Executes the network for a single index in the truth table by calculating the theta values
    * and activation function theta values for nodes at each layer. This method is used during
    * running because it does not store the theta and psi values in arrays
    * @param entry the index in the truth table
    */
   public static void runEntryRun(int entry)
   {
      activationNodes = truthInputs[entry];

      // Calculating the first hidden layer
      for (int k = 0; k < numHiddenNodesFirst; k++)
      {
         double newThetaK = 0.0;
         for (int m = 0; m < numActivationNodes; m++)
         {
            newThetaK += activationNodes[m] * weightsMK[m][k];
         }
         hiddenNodesFirst[k] = activation(newThetaK);
      } // for (int k = 0; k < numHiddenNodesFirst; k++)

      // Calculating the second hidden layer
      for (int j = 0; j < numHiddenNodesSecond; j++)
      {
         double newThetaJ = 0.0;
         for (int k = 0; k < numHiddenNodesFirst; k++)
         {
            newThetaJ += hiddenNodesFirst[k] * weightsKJ[k][j];
         }
         hiddenNodesSecond[j] = activation(newThetaJ);
      } // for (int j = 0; j < numHiddenNodesSecond; j++)

      // Calculating the output layer and psiI
      for (int i = 0; i < numOutputNodes; i++)
      {
         double thetaI = 0.0;
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            thetaI += hiddenNodesSecond[j] * weightsJI[j][i];
         }
         outputNodes[i] = activation(thetaI);
      } // for (int i = 0; i < numOutputNodes; i++)
   } // public static void runEntryRun(int entry)

   /*
    * Executes the network for a single index in the truth table by calculating the theta values
    * and activation function theta values for nodes at each layer. This method is used during
    * training because it stores theta and psi values in arrays
    * @param entry the index in the truth table
    */
   public static void runEntryTrain(int entry)
   {
      activationNodes = truthInputs[entry];

      // Calculating the first hidden layer
      for (int k = 0; k < numHiddenNodesFirst; k++)
      {
         double newThetaK = 0.0;
         for (int m = 0; m < numActivationNodes; m++)
         {
            newThetaK += activationNodes[m] * weightsMK[m][k];
         }
         thetaK[k] = newThetaK;
         hiddenNodesFirst[k] = activation(newThetaK);
      } // for (int k = 0; k < numHiddenNodesFirst; k++)

      // Calculating the second hidden layer
      for (int j = 0; j < numHiddenNodesSecond; j++)
      {
         double newThetaJ = 0.0;
         for (int k = 0; k < numHiddenNodesFirst; k++)
         {
            newThetaJ += hiddenNodesFirst[k] * weightsKJ[k][j];
         }
         thetaJ[j] = newThetaJ;
         hiddenNodesSecond[j] = activation(newThetaJ);
      } // for (int j = 0; j < numHiddenNodesSecond; j++)

      // Calculating the output layer and psiI
      for (int i = 0; i < numOutputNodes; i++)
      {
         double thetaI = 0.0;
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            thetaI += hiddenNodesSecond[j] * weightsJI[j][i];
         }
         outputNodes[i] = activation(thetaI);
         double omegaI = truthOutputs[entry][i] - outputNodes[i];
         psiI[i] = omegaI * activationPrime(thetaI);
      } // for (int i = 0; i < numOutputNodes; i++)
   } // public static void runEntryTrain(int entry)

   /*
    * Executes the network for the entire truth table while printing each truth table entry and
    * its corresponding network output
    */
   public static void run()
   {
      System.out.println("RUNNING RESULTS");
      for (int entry = 0; entry < numTruthEntries; entry++)
      {
         runEntryRun(entry);
         reportRun(entry);
      }
   } // public static void run()

   /*
    * Prints the input, truth output, and network output values for a single test case
    * @param entry the test case that will be printed
    */
   public static void reportRun(int entry)
   {
      // printing the input
//      System.out.print("Input: ");
//      for (int m = 0; m < numActivationNodes; m++)
//      {
//         if (m == numActivationNodes - 1)
//         {
//            System.out.print(truthInputs[entry][m]);
//         }
//         else
//         {
//            System.out.print(truthInputs[entry][m] + ", ");
//         }
//      } // for (int m = 0; m < numActivationNodes; m++)
//
//      // printing the truth output
      System.out.print("True Output: ");
      for (int i = 0; i < numOutputNodes; i++)
      {
         if (i == numOutputNodes - 1)
         {
            System.out.print(truthOutputs[entry][i]);
         }
         else
         {
            System.out.print(truthOutputs[entry][i] + ", ");
         }
      } // for (int i = 0; i < numOutputNodes; i++)

      // printing the network output
      System.out.print(" | Network Output: ");
      for (int i = 0; i < numOutputNodes; i++)
      {
         if (i == numOutputNodes - 1)
         {
            System.out.println(outputNodes[i]);
         }
         else
         {
            System.out.print(outputNodes[i] + ", ");
         }
      } // for (int i = 0; i < numOutputNodes; i++)
   } // public static void reportRun(int entry)

   /*
    * Calculates the network's error by summing the difference between the network output and truth
    * table output squared for each index in the truth table and then dividing the sum by two
    * @param entry the index in the truth table
    * @return the calculated error
    */
   public static double error(int entry)
   {
      double value = 0.0;
      for (int i = 0; i < numOutputNodes; i++)
      {
         value += (truthOutputs[entry][i] - outputNodes[i]) * (truthOutputs[entry][i] - outputNodes[i]);
      }
      return value / 2.0;
   } // public static double error()

   /*
    * Obtains a random number between a specified low value and high value
    * @param lowerLimit the minimum value the random number can take on
    * @param upperLimit the maximum value the random number can take on
    * @return a random number between the low value and high value
    */
   public static double randomNumber(double lowerLimit, double upperLimit)
   {
      return lowerLimit + (upperLimit - lowerLimit) * Math.random();
   }

   /*
    * Assigns each kj and ji weight in the network a random value between a specified low value
    * and high value
    * @param lowerLimit the lowest value that the random weight can take on
    * @param upperLimit the highest value that the random weight can take on
    */
   public static void randomWeights(double lowerLimit, double upperLimit)
   {

      // randomizing the mk weights
      for (int m = 0; m < numActivationNodes; m++)
      {
         for (int k = 0; k < numHiddenNodesFirst; k++)
         {
            weightsMK[m][k] = randomNumber(lowerLimit, upperLimit);
         }
      }
      // randomizing the kj weights
      for (int k = 0; k < numHiddenNodesFirst; k++)
      {
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            weightsKJ[k][j] = randomNumber(lowerLimit, upperLimit);
         }
      }

      // randomizing the ji weights
      for (int j = 0; j < numHiddenNodesSecond; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            weightsJI[j][i] = randomNumber(lowerLimit, upperLimit);
         }
      }
   } // public static void randomWeights(double lowerLimit, double upperLimit)

   /*
    * Uses backpropagation to update the weights for a single index in the truth table and computes
    * the omegaJ and psiI values to assist with the calculations
    */
   public static void deltaWeights()
   {
      // Updating ji weights
      for (int j = 0; j < numHiddenNodesSecond; j++)
      {
         double omegaJ = 0.0;
         for (int i = 0; i < numOutputNodes; i++)
         {
            omegaJ += psiI[i] * weightsJI[j][i];
            weightsJI[j][i] += lambda * hiddenNodesSecond[j] * psiI[i];
         }
         psiJ[j] = omegaJ * activationPrime(thetaJ[j]);
      } // for (int j = 0; j < numHiddenNodesSecond; j++)

      // Updating kj and mk weights
      for (int k = 0; k < numHiddenNodesFirst; k++)
      {
         double omegaK = 0.0;
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            omegaK += psiJ[j] * weightsKJ[k][j];
            weightsKJ[k][j] += lambda * hiddenNodesFirst[k] * psiJ[j];
         }
         double psiK = omegaK * activationPrime(thetaK[k]);

         for (int m = 0; m < numActivationNodes; m++)
         {
            weightsMK[m][k] += lambda * activationNodes[m] * psiK;
         }
      } // for (int k = 0; k < numHiddenNodesFirst; k++)
   } // public static void deltaWeights(int entry)

   /*
    * Begins by assigning the weights random values between a specified lower value and upper
    * value. Updates the weights using gradient descent until either the network error is under
    * a predetermined threshold or the number of iterations exceeds a predetermined maximum amount.
    * Finishes by printing the results of the training to the console.
    */
   public static void train() throws IOException
   {
      // training the network under the given constraints
      time = System.currentTimeMillis();
      error = threshold;
      numIterations = 0;
      while (error >= threshold && numIterations < maxIterations)
      {
         error = 0.0;
         for (int entry = 0; entry < numTruthEntries; entry++)
         {
            runEntryTrain(entry);
            deltaWeights();
            error += error(entry);
         }
         numIterations++;

         // outputting weights at the appropriate intervals
         if (numIterations % intervalValue == 0)
         {
            writeWeights();
         }
         System.out.print("Iteration: " + numIterations + " | ");
         System.out.println("Error: " + error);
      } // while (error >= threshold && numIterations <= maxIterations)

      writeWeights();
      time = System.currentTimeMillis() - time;
      reportTrain();
   } // public static void train()

   /*
    * Prints the results of the previous training to the console. Includes the network
    * configuration, iterations, lambda, random number bounds, error, reason for ending, and
    * weight values. Also runs the network with the optimized weight values and can print the truth
    * table with corresponding network outputs
    */
   public static void reportTrain()
   {
      System.out.println("REPORTING FOR THIS TRAINING CYCLE\n");

      // printing the general network information
      System.out.println("NETWORK PARAMETERS");
      System.out.println("Type: " + numActivationNodes + "-" + numHiddenNodesFirst
              + "-" + numHiddenNodesSecond+ "-" + numOutputNodes);
      System.out.println("Lambda: " + lambda);
      System.out.println("Lower Random Limit: " + lower);
      System.out.println("Upper Random Limit: " + upper);
      System.out.println("Maximum Iterations: " + maxIterations);
      System.out.println("Error Threshold: " + threshold + "\n");

      // printing the statistics of the last training
      System.out.println("TRAINING RESULTS");
      System.out.println("Iterations: " + numIterations);
      System.out.println("Error: " + error);
      String reason = "Iterations Exceed " + maxIterations;
      if (error < threshold)
      {
         reason = "Error Is Under " + threshold;
      }
      System.out.println("Reason For Ending: " + reason);
      System.out.println("Time: " + time + " Milliseconds" + "\n");

      if (printWeights)
      {
         printWeights();
      }

      run();
   } // public static void reportTrain()

   /*
    * Outputs the current weights to the console
    */
   public static void printWeights()
   {
      System.out.println("WEIGHTS");
      // printing the mk weights
      String stringWeightsMK = "MK Weights: (";
      for (int m = 0; m < numActivationNodes; m++)
      {
         for (int k = 0; k < numHiddenNodesFirst; k++)
         {
            if (m == numActivationNodes - 1 && k == numHiddenNodesFirst - 1)
            {
               stringWeightsMK += weightsMK[m][k];
            }
            else
            {
               stringWeightsMK += weightsMK[m][k] + ", ";
            }
         } // for (int k = 0; k < numHiddenNodesFirst; k++)
      } // for (int m = 0; m < numActivationNodes; m++)
      System.out.println(stringWeightsMK + ")");

      // printing the kj weights
      String stringWeightsKJ = "KJ Weights: (";
      for (int k = 0; k < numHiddenNodesFirst; k++)
      {
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            if (k == numHiddenNodesFirst - 1 && j == numHiddenNodesSecond - 1)
            {
               stringWeightsKJ += weightsKJ[k][j];
            }
            else
            {
               stringWeightsKJ += weightsKJ[k][j] + ", ";
            }
         } // for (int j = 0; j < numHiddenNodesSecond; j++)
      } // for (int k = 0; k < numHiddenNodesFirst; k++)
      System.out.println(stringWeightsKJ + ")");

      // printing the ji weights
      String stringWeightsJI = "JI Weights: (";
      for (int j = 0; j < numHiddenNodesSecond; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            if (i == numOutputNodes - 1 && j == numHiddenNodesSecond - 1)
            {
               stringWeightsJI += weightsJI[j][i];
            }
            else
            {
               stringWeightsJI += weightsJI[j][i] + ", ";
            }
         } // for (int i = 0; i < numOutputNodes; i++)
      } // for (int j = 0; j < numHiddenNodesSecond; j++)
      System.out.println(stringWeightsJI + ")" + "\n");
   } // public static void printWeights()

   /*
    * Outputs the current network weights as well as general network information to the file name
    * specified by the configuration file
    */
   public static void writeWeights() throws IOException
   {
      PrintWriter printWriter = new PrintWriter(new FileWriter(outputWeights));

      printWriter.println("Type: " + numActivationNodes + "-" + numHiddenNodesFirst
              + "-" + numHiddenNodesSecond + "-" + numOutputNodes + "\n");

      // printing the mk weights
      printWriter.println("M K Value");
      for (int m = 0; m < numActivationNodes; m++)
      {
         for (int k = 0; k < numHiddenNodesFirst; k++)
         {
            printWriter.println(m + " " + k + " " + weightsMK[m][k]);
         }
      }
      printWriter.println();

      // printing the kj weights
      printWriter.println("K J Value");
      for (int k = 0; k < numHiddenNodesFirst; k++)
      {
         for (int j = 0; j < numHiddenNodesSecond; j++)
         {
            printWriter.println(k + " " + j + " " + weightsKJ[k][j]);
         }
      }
      printWriter.println();

      // printing the ji weights
      printWriter.println("J I Value");
      for (int j = 0; j < numHiddenNodesSecond; j++)
      {
         for (int i = 0; i < numOutputNodes; i++)
         {
            printWriter.println(j + " " + i + " " + weightsJI[j][i]);
         }
      }
      printWriter.close();
   } // public static void writeWeights() throws IOException
} // public class Network