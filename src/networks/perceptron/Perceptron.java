package networks.perceptron;

import networks.Network;
import networks.NetworkModels;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Perceptron implements Network<Double, Double> {
    protected final NetworkModels MODEL = NetworkModels.PERCEPTRON;
    protected Layer[] layers;

    public Perceptron(){}

    @Override
    public int loadFromFile(String fileName) {
        try(BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            int l = -1;
            while((line = br.readLine()) != null){
                if(l < 0) {
                    String[] str = line.split(" ");
                    try {
                        layers = new Layer[Integer.parseInt(str[0])];
                        for (int i = 0; i < layers.length; i++) {
                            if (Integer.parseInt(str[i + 1]) < 0) return 1;
                            layers[i] = new Layer(Integer.parseInt(str[i + 1]));
                        }
                        l++;
                        continue;
                    } catch (NumberFormatException e) {
                        throw new RuntimeException(e);
                    }
                }
                if(l == layers.length - 1) return 1;
                String[] w = line.split(";");
                for (int j = 0; j < layers[l].getLength(); j++) {
                    if (!layers[l].setWeight(w[j].split(" "), j)) return 1;
                }
                if (!layers[l].setWeightB(w[layers[l].getLength()].split(" "))) return 1;
                l++;
            }
            return 0;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int loadFromString(String content) {
        String[] lines = content.split("\n");
        int l = -1;
        for(String line : lines){
            if(l < 0) {
                String[] str = line.split(" ");
                try {
                    layers = new Layer[Integer.parseInt(str[0])];
                    for (int i = 0; i < layers.length; i++) {
                        if (Integer.parseInt(str[i + 1]) < 0) return 1;
                        layers[i] = new Layer(Integer.parseInt(str[i + 1]));
                    }
                    l++;
                    continue;
                } catch (NumberFormatException e) {
                    throw new RuntimeException(e);
                }
            }
            if(l == layers.length - 1) return 1;
            String[] w = line.split(";");
            for (int j = 0; j < layers[l].getLength(); j++) {
                if (!layers[l].setWeight(w[j].split(" "), j)) return 1;
            }
            if (!layers[l].setWeightB(w[layers[l].getLength()].split(" "))) return 1;
            l++;
        }
        return 0;
    }

    @Override
    public void setInput(Double[] input) {
        // Установка входных значений
        layers[0].setInput(input);
    }

    @Override
    public void counting() {
        for(int i = 0; i < layers.length - 1; i++){
            // Установка 0 для следующего слоя
            layers[i+1].setZero();
            // нормализация(активация)
            if(i > 0) layers[i].normalize();
            // Проход по нейронам текущего слоя
            for(int j = 0; j < layers[i].getLength(); j++){
                // Получение нормального значения(0-1) для нейрона
                double curValue = layers[i].getNormResult(j);
                // Проход по нейронам следующего слоя
                for(int k = 0; k < layers[i+1].getLength(); k++){
                    // Добавление к результату произведение значения и веса текущего нейрона + вес нейрона смещения
                    layers[i+1].addResult(k, curValue * layers[i].getWeight(j, k) + layers[i].getWeightB(k));
                }
            }
        }
        // Нормализация выходного слоя
        layers[layers.length - 1].normalize();
    }

    @Override
    public Double[] getOutput() {
        return layers[layers.length - 1].getNormResult();
    }

    @Override
    public NetworkModels getModel() {
        return MODEL;
    }
}
