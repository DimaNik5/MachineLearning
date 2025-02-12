package com.ai.networks.convolutional;

import com.ai.networks.Network;
import com.ai.networks.NetworkModels;
import com.ai.networks.Tokens;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Класс Convolution извлекает признаки из (изображения), в котором реализованы свертка и субдискретизация(пуллинг).
 * Обрабатывает данные типа {@code Matrix} и выдает результат типа {@code Matrix}.
 * Используется сигмоида как функция активации.
 * <p>
 * Примечание: может работать только по заранее подготовленным данных по нейронной сети.
 * </p>
 * @author Никифоров Дмитрий
 * @since 1.1
 */
public class Convolution implements Network<Matrix, Matrix> {
    protected final NetworkModels MODEL;

    {
        MODEL = NetworkModels.CONVOLUTION;
    }

    protected Layer[] layers;

    @Override
    public int loadFromFile(String fileName) {
        try(BufferedReader br = new BufferedReader(new FileReader(fileName))){
            StringBuilder sb = new StringBuilder();
            String s;
            while(!(s = br.readLine()).isEmpty()){
                sb.append(s).append('\n');
            }
            return loadFromString(sb.toString());
        } catch (IOException e) {
            return 1;
        }
    }

    @Override
    public int loadFromString(String content) {
        try{
            String[] l = content.split(Tokens.SEP_OF_LAYERS);
            layers = new Layer[l.length];
            layers[0].loadFirstFromString(l[0]);
            for(int i = 1; i < l.length; i++){
                layers[i].loadFromString(l[i], layers[i - 1].getOut());
            }
        }catch (Exception e) {
            return 1;
        }
        return 0;
    }

    @Override
    public void setInput(Matrix[] input) {
        layers[0].setInputs(input);
    }

    @Override
    public void counting() {
        for (Layer layer : layers) {
            layer.counting();
        }
    }

    @Override
    public Matrix[] getOutput() {
        return layers[layers.length - 1].getOut();
    }

    @Override
    public NetworkModels getModel() {
        return MODEL;
    }
}
