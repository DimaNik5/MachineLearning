package com.ai.networks.convolutional.train.backpropagation;

import com.ai.networks.Teacher;
import com.ai.networks.Tokens;
import com.ai.networks.Training;
import com.ai.networks.convolutional.Convolution;
import com.ai.networks.convolutional.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;

public class TrainConvolutionBackpropagation extends Convolution implements Training<Matrix, Matrix> {

    private TrainLayer[] trainLayers;
    private double speed;
    /**
     * alpha - момент градиентного спуска.
     * Он позволяет выбираться из локальных минимумов.
     */
    private double alpha;
    private double maxWeight;
    private String fileName;

    // {h, w, кол_вход_каналов, (кол_филтров, размер_фильтра,) ...}
    public TrainConvolutionBackpropagation(double[] layers, double speed, double alpha, double maxWeight){
        trainLayers = new TrainLayer[(layers.length - 3) / 2];
        this.speed = speed;
        this.alpha = alpha;
        this.maxWeight = maxWeight;
        trainLayers[0].createNewLayer((int) layers[0], (int) layers[1], (int) layers[2], (int) layers[3], (int) layers[4]);
        for (int i = 1; i < trainLayers.length; i++) {
            trainLayers[i].createNewLayer(trainLayers[i - 1].getOut(), (int) layers[3 + i * 2], (int) layers[4 + i * 2]);
        }
    }

    public TrainConvolutionBackpropagation(String file, double speed, double alpha, double maxWeight){
        this.speed = speed;
        this.alpha = alpha;
        this.maxWeight = maxWeight;
        fileName = file;
        loadFromFile(file);
    }

    @Override
    public void save(String file) {
        fileName = file;
        save();
    }

    @Override
    public void save() {
        if(fileName.isEmpty()) return;
        try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
            StringBuilder cont = trainLayers[0].getContent();
            for (int i = 1; i < trainLayers.length; i++) {
                cont.append(Tokens.SEP_OF_LAYERS);
                trainLayers[i].getContent(cont);
            }
            bw.write(cont.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public double[] getDeltaOfInputLayer() {
        int l = 0;
        for(Matrix m : trainLayers[0].getDeltasOfInput()){
            l += m.getH() * m.getW();
        }
        double[] res = new double[l];
        int k = 0;
        for(Matrix m : trainLayers[0].getDeltasOfInput()){
            double[][] mat = m.getMatrix();
            for (double[] doubles : mat) {
                for (double aDouble : doubles) {
                    res[k++] = aDouble;
                }
            }
        }
        return res;
    }

    @Override
    public void setDeltaOfOutputLayer(double[] deltas) {
        int h = trainLayers[0].getOut()[0].getH();
        int w = trainLayers[0].getOut()[0].getW();
        h = h /2 + h % 2;
        w = w / 2 + w % 2;
        trainLayers[0].setDeltas(Matrix.getMatricesFromArray(deltas, trainLayers[0].getOut().length, h, w));
        trainLayers[0].correcting(speed, alpha, maxWeight);
        for (int i = 1; i < trainLayers.length; i++) {
            trainLayers[i].setDeltas(trainLayers[i - 1].getDeltasOfInput());
            trainLayers[i].correcting(speed, alpha, maxWeight);
        }
    }

    @Override
    public double[] getErrors() {
        return null;
    }
    @Override
    public int training(Teacher<Matrix, Matrix> dataset, int epochs) {
        return 1;
    }
    @Override
    public int printingStatus(PrintStream printStream) {
        return 1;
    }
    @Override
    public void stop() {

    }
    @Override
    public Thread getThread() {
        return null;
    }
    @Override
    public boolean isTraining() {
        return false;
    }

}
