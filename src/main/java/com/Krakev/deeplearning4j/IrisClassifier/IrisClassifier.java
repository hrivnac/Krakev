package com.Krakev.deeplearning4j.IrisClassifier;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
//import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

//import org.deeplearning4j.api.storage.StatsStorage;
//import org.deeplearning4j.examples.quickstart.features.userinterface.util.UIExampleUtils;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.ui.api.UIServer;
//import org.deeplearning4j.ui.model.stats.StatsListener;
//import org.deeplearning4j.ui.model.storage.FileStatsStorage;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.ScatterPlot;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import tech.tablesaw.api.Table;

import java.io.File;


import java.io.File;
import java.io.IOException;

public class IrisClassifier {

  public static void main(String[] args) throws IOException, InterruptedException {
  
    DataSet allData;
        
    try (RecordReader recordReader = new CSVRecordReader(1, ',')) {
      //recordReader.initialize(new FileSplit(new ClassPathResource("iris.cvs").getFile()));
      recordReader.initialize(new FileSplit(new File("src/main/resources//iris.txt")));
      DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
      allData = iterator.next();
      }
      
    allData.setLabelNames(Arrays.asList(new String[]{"sepal.length", "sepal.width", "petal.length", "petal.width", "variety"}));
    allData.shuffle(42);
    
    DataNormalization normalizer = new NormalizerStandardize();
    normalizer.fit(allData);
    normalizer.transform(allData);

    SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
    DataSet trainingData = testAndTrain.getTrain();
    DataSet testData = testAndTrain.getTest();

    MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                                                                      //.iterations(1000)
                                                                      .activation(Activation.TANH)
                                                                      .weightInit(WeightInit.XAVIER)
                                                                      //.regularization(true)
                                                                      //.learningRate(0.1)
                                                                      .l2(0.0001)
                                                                      .list()
                                                                      .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT)
                                                                                                        .nOut(3)
                                                                                                        .build())
                                                                      .layer(1, new DenseLayer.Builder().nIn(3)
                                                                                                        .nOut(3)
                                                                                                        .build())
                                                                      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                                                                                                                                                         .nIn(3)
                                                                                                                                                         .nOut(CLASSES_COUNT)
                                                                                                                                                         .build())
                                                                      .backpropType(BackpropType.Standard)
                                                                      //.pretrain(false)
                                                                      .build();

    MultiLayerNetwork model = new MultiLayerNetwork(configuration);
    model.init();
    model.fit(trainingData);
    
    //UIServer uiServer = UIServer.getInstance();
    //StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
    //int listenerFrequency = 1;
    //model.setListeners(new StatsListener(statsStorage, listenerFrequency));
    //uiServer.attach(statsStorage);
    
    
    INDArray output = model.output(testData.getFeatures());
    
    
    Evaluation eval = new Evaluation(CLASSES_COUNT);
    eval.eval(testData.getLabels(), output);
    System.out.println(eval.stats());
    Table table = createTableFromNetwork(model);    
    //Plot.show(ScatterPlot.create("Test", table, "Layer", "Weight"));
    
    }

  public static Table createTableFromNetwork(MultiLayerNetwork network) {
    List<String> layerNames = new ArrayList<>();
    List<Double> weights    = new ArrayList<>();
    List<Double> biases     = new ArrayList<>();
    for (int i = 0; i < network.getnLayers(); i++) {
      Layer layer = network.getLayer(i);
      INDArray weightArray = layer.getParam("W");
      //INDArray biasArray = layer.getParam("b");
      for (int j = 0; j < weightArray.length(); j++) {
        layerNames.add("L" + i + "W");
        weights.add(weightArray.getDouble(j));
        }        
      //for (int j = 0; j < biasArray.length(); j++) {
      //    layerNames.add("Layer " + i + " Bias");
      //    biases.add(biasArray.getDouble(j));
      //}
      }
    StringColumn layerColumn = StringColumn.create("Layer", layerNames);
    DoubleColumn weightColumn = DoubleColumn.create("Weight", weights);
    //DoubleColumn biasColumn = DoubleColumn.create("Bias", biases);
    Table table = Table.create("Network Parameters").addColumns(layerColumn, weightColumn);
    return table;
    }

  private static final int CLASSES_COUNT = 3;

  private static final int FEATURES_COUNT = 4;  
    
  }
