package com.ai.networks.perceptron.train.teacher;

import com.ai.networks.Teacher;
import com.ai.networks.Tokens;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Класс PerceptronTeacher хранит входые и ожидаемые значения значения
 * @author Никифоров Дмитрий
 * @since 1.0
 */
public class PerceptronTeacher implements Teacher<Double, Double> {
    /**
     * inputs - список массивов входных значений типа {@code Double}
     */
    private List<Double[]> inputs;

    /**
     * outputs - список массивов ожидаемых значений типа {@code Double}
     */
    private List<Double[]>  outputs;

    /**
     * loadData - индивидуальный метод для загрузки датасета из файлов
     * @param fileIn {@code String} - путь до файла, содержащий входные значения
     * @param fileOut {@code String} - путь до файла, содержащий ожидаемые значения
     * @return результат загрузки. При успешной - 0. При неудаче - 1.
     */
    public int loadData(String fileIn, String fileOut){
        int c = 0;
        try(BufferedReader bfi = new BufferedReader(new FileReader(fileIn)); BufferedReader bfo = new BufferedReader(new FileReader(fileOut))){
            while(bfi.readLine() != null && bfo.readLine() != null){
                c++;
            }
        }catch (Exception e){
            return 1;
        }
        inputs = new ArrayList<>(c);
        outputs = new ArrayList<>(c);
        try(BufferedReader bfi = new BufferedReader(new FileReader(fileIn)); BufferedReader bfo = new BufferedReader(new FileReader(fileOut))){
            String li, lo;
            int j = 0;
            while((li = bfi.readLine()) != null && (lo = bfo.readLine()) != null){
                String[] str = li.split(Tokens.SEP_IN_LIST);
                if(j != 0) {
                    if(inputs.get(j - 1).length != str.length) continue;
                }
                String[] str1 = lo.split(Tokens.SEP_IN_LIST);
                if(j != 0) {
                    if(outputs.get(j - 1).length != str1.length) continue;
                }
                inputs.set(j, new Double[str.length]);
                for(int i = 0; i < str.length; i++){
                    inputs.get(j)[i] = Double.parseDouble(str[i]);
                }

                outputs.set(j, new Double[str.length]);
                for(int i = 0; i < str1.length; i++){
                    outputs.get(j)[i] = Double.parseDouble(str1[i]);
                }
                j++;
            }
        }catch (Exception e){
            return 1;
        }
        return 0;
    }

    @Override
    public int setData(Double[][] inputs, Double[][] outputs) {
        if(inputs.length == outputs.length && inputs.length > 0){
            this.inputs = new ArrayList<>(inputs.length);
            this.outputs = new ArrayList<>(inputs.length);
            this.inputs.addAll(Arrays.asList(inputs));
            this.outputs.addAll(Arrays.asList(outputs));
            return 0;
        }
        return 1;
    }

    @Override
    public int addData(Double[][] inputs, Double[][] outputs) {
        if(inputs.length == outputs.length && inputs.length > 0){
            if(this.inputs == null) {
                this.inputs = new ArrayList<>(inputs.length);
                this.outputs = new ArrayList<>(inputs.length);
            }
            this.inputs.addAll(Arrays.asList(inputs));
            this.outputs.addAll(Arrays.asList(outputs));
            return 0;
        }
        return 1;
    }

    @Override
    public int addData(Double[] inputs, Double[] outputs) {
        if(inputs.length > 0 && outputs.length > 0){
            if(this.inputs == null) {
                this.inputs = new ArrayList<>();
                this.outputs = new ArrayList<>();
            }
            this.inputs.add(inputs);
            this.outputs.add(inputs);
            return 0;
        }
        return 1;
    }

    @Override
    public int setData(List<Double[]> inputs, List<Double[]> outputs) {
        if(inputs == null || outputs == null || inputs.size() != outputs.size() || inputs.size() == 0) return 1;
        this.inputs = new ArrayList<>(inputs);
        this.outputs = new ArrayList<>(outputs);
        return 0;
    }

    @Override
    public int addData(List<Double[]> inputs, List<Double[]> outputs) {
        if(inputs == null || outputs == null || inputs.size() != outputs.size() || inputs.size() == 0) return 1;
        this.inputs.addAll(inputs);
        this.outputs.addAll(outputs);
        return 0;
    }

    @Override
    public List<Double[]> getIn() {
        return inputs;
    }

    @Override
    public List<Double[]> getOut() {
        return outputs;
    }

    @Override
    public boolean checkData() {
        return inputs != null && outputs != null && inputs.size() == outputs.size() && inputs.size() > 0;
    }
}
