/*
 * Linear Genetic Programming
 *
 * Jorge Martinez-Gil: A Comparative Study of Ensemble Techniques Based on Genetic Programming: 
 * A Case Study in Semantic Similarity Assessment. Int. J. Softw. Eng. Knowl. Eng. 33(2): 289-312 (2023)
 *
 * @author: Jorge Martinez-Gil
 */

package lgp;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.github.chen0040.gp.lgp.LGP;
import com.github.chen0040.gp.commons.BasicObservation;
import com.github.chen0040.gp.commons.Observation;
import com.github.chen0040.gp.lgp.program.Program;
import com.github.chen0040.gp.lgp.program.operators.*;

public class lgp {

    private static final int FEATURE_COUNT = 5;
    private static final String DEFAULT_DATASET = "mc";

    private static List<Observation> generate(Path file) {
        List<Observation> result = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(file)) {
            String data = reader.readLine();
            while ((data = reader.readLine()) != null) {
                if (data.trim().length() == 0) {
                    continue;
                }

                final String[] line = data.split(",");
                if (line.length == FEATURE_COUNT + 1) {
                    try {
                        final double label = Double.parseDouble(line[0].trim());
                        Observation observation = new BasicObservation(FEATURE_COUNT, 1);

                        for (int i = 0; i < FEATURE_COUNT; i++) {
                            observation.setInput(i, Double.parseDouble(line[i + 1].trim()));
                        }
                        observation.setOutput(0, label);

                        result.add(observation);
                    } catch(final NumberFormatException e) {
                        System.out.println(e);
                    }
                }
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
        return result;
    }

    private static Path datasetPath(String datasetName, String split) {
        String fileName = datasetName + "-" + split + ".txt";
        Path fromProjectRoot = Paths.get("datasets", fileName);
        if (Files.exists(fromProjectRoot)) {
            return fromProjectRoot;
        }
        return Paths.get("..", "..", "datasets", fileName).normalize();
    }
	
	   
		public static double getPearson(double[] scores1, double[] scores2) {
			double result = 0;
			double sum_sq_x = 0;
			double sum_sq_y = 0;
			double sum_coproduct = 0;
			double mean_x = scores1[0];
			double mean_y = scores2[0];
			for (int i = 2; i < scores1.length + 1; i += 1) {
				double sweep = Double.valueOf(i - 1) / i;
				double delta_x = scores1[i - 1] - mean_x;
				double delta_y = scores2[i - 1] - mean_y;
				sum_sq_x += delta_x * delta_x * sweep;
				sum_sq_y += delta_y * delta_y * sweep;
				sum_coproduct += delta_x * delta_y * sweep;
				mean_x += delta_x / i;
				mean_y += delta_y / i;
			}
			double pop_sd_x = (double) Math.sqrt(sum_sq_x / scores1.length);
			double pop_sd_y = (double) Math.sqrt(sum_sq_y / scores1.length);
			double cov_x_y = sum_coproduct / scores1.length;
			result = cov_x_y / (pop_sd_x * pop_sd_y);
			
			if (Double.isNaN(result))
				return 0;
			else
				return result;
		}
		
		
		private static double getSpearman(double [] X, double [] Y) {
		       
			try {
			/* Error check */
		        if (X == null || Y == null || X.length != Y.length) {
		            return (Double) null;
		        }
		        
		        /* Create Rank arrays */
		        int [] rankX = getRanks(X);
		        int [] rankY = getRanks(Y);

		        /* Apply Spearman's formula */
		        int n = X.length;
		        double numerator = 0;
		        for (int i = 0; i < n; i++) {
		            numerator += Math.pow((rankX[i] - rankY[i]), 2);
		        }
		        numerator *= 6;
		        return 1 - numerator / (n * ((n * n) - 1));
			}
			catch (Exception e) {
				return -999;
			}
		    }
		    
		    /* Returns a new (parallel) array of ranks. Assumes unique array values */
		    public static int[] getRanks(double [] array) {
		        int n = array.length;
		        
		        /* Create Pair[] and sort by values */
		        Pair [] pair = new Pair[n];
		        for (int i = 0; i < n; i++) {
		            pair[i] = new Pair(i, array[i]);
		        }
		        Arrays.sort(pair, new PairValueComparator());

		        /* Create and return ranks[] */
		        int [] ranks = new int[n];
		        int rank = 1;
		        for (Pair p : pair) {
		            ranks[p.index] = rank++;
		        }
		        return ranks;
		    }
		

	public static void main(String[] args) {
		
		double[] source;
		double[] target;
		String datasetName = args.length > 0 ? args[0] : DEFAULT_DATASET;
		
		List<Observation> trainingData = generate(datasetPath(datasetName, "training"));
		List<Observation> testingData = generate(datasetPath(datasetName, "validation"));
		
		/*CollectionUtils.shuffle(data);
		TupleTwo<List<Observation>, List<Observation>> split_data = CollectionUtils.split(data, 0.8);
		List<Observation> trainingData = split_data._1();
		List<Observation> testingData = split_data._2();*/
		
		LGP lgp = LGP.defaultConfig();
	      lgp.getOperatorSet().addAll(new Plus(), new Minus(), new Divide(), new Multiply(), new Sine(), new Cosine());
	      lgp.getOperatorSet().addIfLessThanOperator();
	      lgp.addConstants(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0);
		
		lgp.setRegisterCount(FEATURE_COUNT * 3);
		lgp.setCostEvaluator((program, observations)->{
		 int j = 0;
		 double[] s = new double [observations.size()];
	     double[] t = new double [observations.size()];
		 for(Observation observation : observations){
		    program.execute(observation);
			double predicted = observation.getPredictedOutput(0);
			double actual = observation.getOutput(0);
			s[j] = predicted;
			t[j] = actual;
			j++; 
		 }

		 return -getSpearman(s,t);
		});

		lgp.setDisplayEvery(50); // to display iteration results every 10 generation

		Program program = lgp.fit(trainingData);
		System.out.println(program);
		
		int i = 0;
		
		source = new double [testingData.size()];
		target = new double [testingData.size()];
		
		for(Observation observation : testingData) {
			 program.execute(observation);
			 double predicted = observation.getPredictedOutput(0);
			 double actual = observation.getOutput(0);

			 System.out.println (predicted + " " + actual);
			 
			 source[i] = predicted;
			 target[i] = actual;
			 i++;
		}
		
		System.out.println ("Correl: " + getSpearman(target,source));
			
	}

}
