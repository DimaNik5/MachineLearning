package networks.perceptron.train;

import networks.Teacher;
import networks.Training;
import networks.perceptron.Layer;
import networks.perceptron.Perceptron;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainPerceptron extends Perceptron implements Training {

    private TrainLayer[] trainLayers;
    private double speed;
    private double alpha;
    private double maxWeight;
    private String fileName;

    private double middleError = 0, maxError = 0;

    private boolean isTraining = false, isPrinting = false;
    private int countDone;
    private int epochs = 1;
    private List<Double[]> in;
    private List<Double[]> out;
    private final Thread training;
    private final Thread printing;

    private final PrintStream printStream;


    {
        try {
            printStream = new PrintStream(System.out, true, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }

        training = new Thread(() -> {
            countDone = 0;
            double mide, maxe;
            // Тренировка по эпохам
            for (int k = 0; k < epochs; k++){
                mide = 0;
                maxe = 0;
                // Проход по тестам
                for (int i = 0; i < in.size(); i++) {
                    setInput(in.get(i));
                    // Подсчет результата
                    counting();
                    double t = Math.abs(out.get(i)[0] - getOutput()[0]);
                    mide += t;
                    maxe = Math.max(t, maxe);
                    // Корректировка весов
                    correction(out.get(i));
                }
                mide /= in.size();
                middleError = mide;
                maxError = maxe;
                countDone++;
                if(!isTraining) break;
            }
            isPrinting = false;
        });

        printing = new Thread(() -> {

            String proc = "Пройдено эпох ";
            String me = "\nСредняя ошибка :";
            String maxe = "\nМаксимальная ошибка: ";
            StringBuilder res;
            boolean last = true;
            while(isPrinting || last){
                System.out.println(new String(new char[25]).replace("\0", "\r\n"));
                res = new StringBuilder(proc + countDone + '/' + epochs + " [");
                int part = (countDone * 10 / epochs) % 100;
                res.append("+".repeat(Math.max(0, part)));
                res.append("-".repeat(Math.max(0, 10 - part)));
                res.append(']');
                res.append(me).append(String.format("%.3f", middleError));
                res.append(maxe).append(String.format("%.3f", maxError));
                printStream.println(res);
                last = isPrinting;
                try {
                    Thread.sleep(400);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    public TrainPerceptron(int[] lenLayer, double speed, double alpha, double maxWeight){
        int len = lenLayer.length;
        if(len < 2) return;
        this.speed = speed;
        this.maxWeight = maxWeight;
        this.alpha = alpha;
        trainLayers = new TrainLayer[len];
        for(int i = 0; i < len; i++){
            if(lenLayer[i] < 0) return;
            trainLayers[i] = new TrainLayer(lenLayer[i], (i < len - 1 ? lenLayer[i + 1] : 0));
        }
        layers = trainLayers;
    }

    public TrainPerceptron(String fileName, double speed, double alpha, double maxWeight){
        this.speed = speed;
        this.maxWeight = maxWeight;
        this.alpha = alpha;
        this.fileName = fileName;
        loadFromFile(fileName);
    }


    @Override
    public int training(Teacher dataset, int epochs) {
        if(!dataset.checkData()) return 1;

        this.epochs = epochs;

        in = new ArrayList<>();

        for(int i = 0; i < dataset.getIn().length; i++){
            if(dataset.getIn()[i].length != trainLayers[0].getLength()) return 1;
            in.add(Arrays.stream(dataset.getIn()[i]).toArray(Double[]::new));
        }

        out = new ArrayList<>();

        for(int i = 0; i < dataset.getOut().length; i++){
            if(dataset.getOut()[i].length != trainLayers[trainLayers.length - 1].getLength()) return 1;
            out.add(Arrays.stream(dataset.getOut()[i]).toArray(Double[]::new));
        }

        // in.forEach(e -> System.out.println(Arrays.toString(e)));
        // out.forEach(e -> System.out.println(Arrays.toString(e)));

        if(!training.isAlive()){
            isTraining = true;
            training.start();
            return 0;
        }
        return 1;
    }

    @Override
    public int printingStatus() {
        if(!printing.isAlive()){
            isPrinting = true;
            printing.start();
            return 0;
        }
        return 1;
    }

    @Override
    public void stop() {
        if(isTraining) isTraining = false;
    }

    @Override
    public Thread getThread() {
        return training;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void save(String file) {

        if(fileName == null || !fileName.equals(file)) fileName = file;
        try(FileWriter fw = new FileWriter(fileName)) {
            // Запись количества слоев
            fw.write(trainLayers.length + " ");
            // Запись количества нейронов в каждом слое
            for(TrainLayer layerTrain : trainLayers){
                fw.write(layerTrain.getLength() + " ");
            }
            fw.write('\n');

            // Проход по всем слоям, кроме выходного(нет весов)
            for(int k = 0; k < trainLayers.length - 1; k++){
                // Проход повсем нейронам слоя
                for(int i = 0; i < trainLayers[k].getLength(); i++){
                    // Проход повсем нейронам следующего слоя
                    for(int j = 0; j < trainLayers[k + 1].getLength(); j++){
                        // Запись весов от i к j нейрону
                        fw.write(trainLayers[k].getWeight(i, j) + " ");
                    }
                    // Знак препинания
                    // TODO сделать константы
                    fw.write(';');
                }
                // Запись весов нейрона смещения
                for(int j = 0; j < trainLayers[k + 1].getLength(); j++){
                    fw.write(trainLayers[k].getWeightB(j) + " ");
                }
                fw.write('\n');
            }
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void save() {
        if(!fileName.isEmpty()){
            save(fileName);
        }
    }

    @Override
    public double[] getErrors() {
        return new double[]{middleError, maxError};
    }

    private void correction(Double[] ideal){
        Double[] curIdeal = ideal;
        // Получение дельт(разницы от идеала) для каждого слоя с передачей результатов к предыдущему
        for(int i = trainLayers.length - 1; i > 0; i--){
            curIdeal = trainLayers[i].setDelta(curIdeal, i == trainLayers.length - 1);
        }
        boolean f = false; // Превышение максимального веса
        // Установка новых весов
        for(int i = 0; i < trainLayers.length - 1; i++){
            f = f || (Math.abs(trainLayers[i].setDeltaWeight(trainLayers[i + 1].getDeltas(), speed, alpha)) > maxWeight);
        }
        if(f){
            // Пропорциональное деление весов
            for(int i = 0; i < trainLayers.length - 1; i++){
                trainLayers[i].divWeight(maxWeight / 2);
            }
        }

    }

    @Override
    public int loadFromFile(String fileName) {
        try(BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            int l = -1;
            while((line = br.readLine()) != null){
                if(l < 0) {
                    String[] str = line.split(" ");
                    try {
                        trainLayers = new TrainLayer[Integer.parseInt(str[0])];
                        for (int i = 0; i < trainLayers.length; i++) {
                            if (Integer.parseInt(str[i + 1]) <= 0) return 1;
                            trainLayers[i] = new TrainLayer(Integer.parseInt(str[i + 1]), (i < trainLayers.length - 1 ? Integer.parseInt(str[i + 2]) : 0));
                        }
                        l++;
                        continue;
                    } catch (NumberFormatException e) {
                        throw new RuntimeException(e);
                    }
                }
                if(l == trainLayers.length - 1) return 1;
                String[] w = line.split(";");
                for (int j = 0; j < trainLayers[l].getLength(); j++) {
                    if (!trainLayers[l].setWeight(w[j].split(" "), j)) return 1;
                }
                if (!trainLayers[l].setWeightB(w[trainLayers[l].getLength()].split(" "))) return 1;
                l++;
            }
            layers = trainLayers;
            return 0;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int loadFromString(String content) {
        String[] lines = content.split("\n");
        System.out.println(lines[0]);
        int l = -1;
        for(String line : lines){
            if(l < 0) {
                String[] str = line.split(" ");
                try {
                    trainLayers = new TrainLayer[Integer.parseInt(str[0])];
                    for (int i = 0; i < trainLayers.length; i++) {
                        if (Integer.parseInt(str[i + 1]) <= 0) return 1;
                        trainLayers[i] = new TrainLayer(Integer.parseInt(str[i + 1]), (i < trainLayers.length - 1 ? Integer.parseInt(str[i + 2]) : 0));
                    }
                    l++;
                    continue;
                } catch (NumberFormatException e) {
                    throw new RuntimeException(e);
                }
            }
            if(l == trainLayers.length - 1) return 1;
            String[] w = line.split(";");
            for (int j = 0; j < trainLayers[l].getLength(); j++) {
                if (!trainLayers[l].setWeight(w[j].split(" "), j)) return 1;
            }
            if (!trainLayers[l].setWeightB(w[trainLayers[l].getLength()].split(" "))) return 1;
            l++;
        }
        layers = trainLayers;
        return 0;
    }
}
