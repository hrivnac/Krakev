
package com.Krakev.deeplearning4j.Utils;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DataOrganizer {

  public DataOrganizer(String baseDirName) throws Exception {
    baseDir          = new File(baseDirName);
    baseTrainDir     = new File(baseDir,      "train");
    featuresDirTrain = new File(baseTrainDir, "features");
    labelsDirTrain   = new File(baseTrainDir, "labels");
    baseTestDir      = new File(baseDir,      "test");
    featuresDirTest  = new File(baseTestDir,  "features");
    labelsDirTest    = new File(baseTestDir,  "labels");
    }

  public String getExampleData() throws Exception {
    String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
    String data = IOUtils.toString(new URL(url), (Charset) null);
    return data;
    }  
  public void prepareData(String data,
                          int    blockSize,
                          int    trainSize) throws Exception {
    
    String[] lines = data.split("\n");
    
    // Create directories
    baseDir.mkdir();
    baseTrainDir.mkdir();
    featuresDirTrain.mkdir();
    labelsDirTrain.mkdir();
    baseTestDir.mkdir();
    featuresDirTest.mkdir();
    labelsDirTest.mkdir();
    
    int lineCount = 0;
    List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
    for (String line : lines) {
      String transposed = line.replaceAll(" +", "\n");
      // Labels: first blockSize lines are label 0, second blockSize lines are label 1, and so on
      contentAndLabels.add(new Pair<>(transposed, lineCount++ / blockSize));
      }
    
    // Randomize
    Collections.shuffle(contentAndLabels, new Random(12345));
    
    // trainSize to train, rest for test
    int nTrain = trainSize;
    int trainCount = 0;
    int testCount = 0;
    for (Pair<String, Integer> p : contentAndLabels) {
      File outPathFeatures;
      File outPathLabels;
      if (trainCount < nTrain) {
        outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
        outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
        trainCount++;
        }
      else {
        outPathFeatures = new File(featuresDirTest, testCount + ".csv");
        outPathLabels = new File(labelsDirTest, testCount + ".csv");
        testCount++;
        }
      
      FileUtils.writeStringToFile(outPathFeatures, p.getFirst(), (Charset) null);
      FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString(), (Charset) null);
      }
    }
    
  public File baseDir;
  public File baseTrainDir;
  public File featuresDirTrain;
  public File labelsDirTrain;
  public File baseTestDir;
  public File featuresDirTest;
  public File labelsDirTest;
        
  private static final Logger log = LoggerFactory.getLogger(DataOrganizer.class);
    
  }
