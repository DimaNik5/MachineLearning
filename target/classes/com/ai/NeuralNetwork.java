package com.ai;

import com.ai.networks.Network;
import com.ai.networks.Teacher;
import com.ai.networks.Training;
import com.ai.networks.perceptron.train.backpropagation.TrainPerceptronBackpropagation;
import com.ai.networks.perceptron.train.teacher.PerceptronTeacher;

/**
 * Класс NeuralNetwork предназначен для создания нейронной сети.<br>
 * Проект создан 17 декабря 2024 года.
 * @author Никифоров Дмитрий
 * @version 1.0
 */
public class NeuralNetwork {
    /* public static void main(String[] args) {
        String filename = System.getProperty("user.dir") + "/weight.txt";
        //TrainPerceptronBackpropagation tp = new TrainPerceptronBackpropagation(filename, 0.07, 0.03, 10);
        TrainPerceptronBackpropagation tp = new TrainPerceptronBackpropagation(new int[]{2, 8, 8, 8, 1}, 0.07, 0.03, 10);
        Teacher<Double, Double> t = new PerceptronTeacher();
        t.setData(new Double[][]{
                {0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}
                },
                new Double[][]{
                        {0.0}, {1.0}, {1.0}, {0.0}
        });

        tp.training(t, 100000);
        // Для корректного отображения в консоле IDE можно передать null
        tp.printingStatus(System.out);
        try {
            tp.getThread().join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        tp.save(filename);
    }*/

    // TODO методы для создания сложных сетей (ансамбль, сверточная)
}