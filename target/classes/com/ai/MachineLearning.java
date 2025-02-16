package com.ai;

import com.ai.networks.convolutional.Matrix;
import com.ai.networks.convolutional.train.backpropagation.TrainConvolutionBackpropagation;
import com.ai.networks.perceptron.train.backpropagation.TrainPerceptronBackpropagation;

import java.io.*;
import java.util.Arrays;

/**
 * Класс MachineLearning предназначен для удобства создания моделей машинного обучения.<br>
 * Проект создан 17 декабря 2024 года.
 * <p>
 * Примечание: данная библиотека разработана на 11 версией языка Java.
 * </p>
 * @author Никифоров Дмитрий
 * @version 1.1
 */
public class MachineLearning {

    // TODO методы для создания сложных сетей (ансамбль, сверточная)

    public static void main(String[] args) throws IOException {
        boolean NEW = false;
        String path = "C:\\Users\\Президент\\Desktop\\test/";
        Double[] answer = new Double[]{(double) 0, (double) 1, (double) 1, (double) 0, (double) 1, (double) 0, (double) 0, (double) 0, (double) 1};
        TrainConvolutionBackpropagation cnn;
        TrainPerceptronBackpropagation pn;
        if(NEW){
            cnn = new TrainConvolutionBackpropagation(new int[]{100, 100, 3, 6, 3, 4, 3, 2, 3}, 0.02, 0.05, 30);
            pn = new TrainPerceptronBackpropagation(new int[]{11 * 11 * 2, 50, 25, 4, 1}, 0.02, 0.05, 30);
        }
        else {
            BufferedReader bf = new BufferedReader(new FileReader("CNN.txt"));
            StringBuilder s = new StringBuilder(bf.readLine());
            String t = bf.readLine();
            while (t != null) {
                s.append("\n");
                s.append(t);
                t = bf.readLine();
            }
            bf.close();
            String[] n = s.toString().split("q");
            cnn = new TrainConvolutionBackpropagation(n[0], 0.2, 0.05, 30);
            pn = new TrainPerceptronBackpropagation(new int[]{1, 1}, 0.2, 0.05, 30);
            pn.loadFromString(n[1]);
        }
        for (int j = 0; j < 100; j++) {

            for (int i = 0; i < answer.length; i++) {
                cnn.setInput(Matrix.getFromImage(path + i + ".png", Matrix.RGB_CHANNELS));
                cnn.counting();
                pn.setInput(parseMatrices(cnn.getOutput()));
                pn.counting();
                System.out.println(Arrays.toString(pn.getOutput()) + ", ideal: " + answer[i]);
                pn.setDeltaOfOutputLayer(new double[]{answer[i]});
                cnn.setDeltaOfOutputLayer(pn.getDeltaOfInputLayer());
            }
            System.out.println();
        }

        String r = cnn.getContent() + "q" +
                pn.getContent();
        BufferedWriter bw = new BufferedWriter(new FileWriter("CNN.txt"));
        bw.write(r);
        bw.close();
    }

    private static Double[] parseMatrices(Matrix[] m){
        Double[] res = new Double[m.length * m[0].getW() * m[0].getH()];
        for (int k = 0; k < m.length; k++) {
            for (int i = 0; i < m[0].getH(); i++) {
                for (int j = 0; j < m[0].getW(); j++) {
                    res[k * m[0].getW() * m[0].getH() + i * m[0].getW() + j] = m[k].getCell(i, j);
                }
            }
        }
        return res;
    }
}