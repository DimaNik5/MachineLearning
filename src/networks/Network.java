package networks;

public interface Network<IN, OUT> {

    int loadFromFile(String fileName);

    int loadFromString(String content);

    void setInput(IN[] input);

    void counting();

    OUT[] getOutput();

    NetworkModels getModel();
}
