package weka.clusterers;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.Serializable;
import java.util.*;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.IntStream;

public class CLARANS<T> extends RandomizableClusterer implements OptionHandler, Randomizable {

    protected List<int[]> clusters;
    protected List<int[]> current;
    protected List<int[]> belong;
    protected List<int[]> optimal_medoids;
    protected double[] optimal_estimation;
    protected int number_clusters;
    protected int num_local;
    protected int maximumNeighbor;

    /**
     * replace missing values in training instances
     */
    protected ReplaceMissingValues ReplaceMissingVal;

    public CLARANS() {
        number_clusters = 3;
        num_local = 6;
        maximumNeighbor = 4;

    }
    // constuctor
    public <T> CLARANS(double distortion, T[] medoids, int[] y, Distance<T> distance) {
        super();

    }

    /*   calculate a distance measure between two objects */
    public interface Distance<T> extends ToDoubleBiFunction<T,T>, Serializable {
        double d(T x, T y);
    }

    /* Clustering data into k clusters.
    * n = number of data
    * k = number clusters.
     * */
    public static <T> CLARANS<Object> fit(T[] data, Distance<T> distance, int k, int maximumNeighbor) {
        if (maximumNeighbor <= 0) {
            throw new IllegalArgumentException("Choose valid maximum neighbor > 0");
        }
        int n = data.length;

        if (k >= n) {
            throw new IllegalArgumentException("Number of clusters is too large:" + k + ". it should be >=" + n) ;
        }

        if (maximumNeighbor > n) {
            throw new IllegalArgumentException("Maximum Neighbor should be > " + n);
        }

        int minmax = 100;
        if (k * (n - k) < minmax) {
            minmax = k * (n - k);
        }
        if (maximumNeighbor < minmax) {
            maximumNeighbor = minmax;
        }

        @SuppressWarnings("unchecked")
        T[] medoids = (T[]) java.lang.reflect.Array.newInstance(data.getClass().getComponentType(), k);
        T[] __Medoids = medoids.clone();
        int[] y = new int[n];
        int[] __Y = new int[n];
        double[] __D = new double[n];

        // clarans need a initial clustering configuration as a seed.
        double[] d = seed(data, medoids, y, distance);
        double distortion = Calculate.sum(d);

        System.arraycopy(medoids, 0, __Medoids, 0, k);
        System.arraycopy(y, 0, __Y, 0, n);
        System.arraycopy(d, 0, __D, 0, n);

        for (int neighborCount = 1; neighborCount <= maximumNeighbor; neighborCount++) {
            double randomNeighborDistortion = getRandomNeighbor(data, __Medoids, __Y, __D, distance);
            if (randomNeighborDistortion < distortion) {
                neighborCount = 0;
                distortion = randomNeighborDistortion;
                System.arraycopy(__Medoids, 0, medoids, 0, k);
                System.arraycopy(__Y, 0, y, 0, n);
                System.arraycopy(__D, 0, d, 0, n);
            } else {
                System.arraycopy(medoids, 0, __Medoids, 0, k);
                System.arraycopy(y, 0, __Y, 0, n);
                System.arraycopy(d, 0, __D, 0, n);
            }
        }
        return new CLARANS(distortion, medoids, y, distance);
    }
    private static <T> double getRandomNeighbor(T[] data, T[] medoids, int[] y, double[] d, Distance<T> distance) {
        return 0;
    }

    //seed
    private static <T> double[] seed(T[] data, T[] medoids, int[] y, Distance<T> distance) {
        int n = data.length;
        int k = medoids.length;
        double[] d = new double[n];
        medoids[0] = data[Calculate.randomInt(n)];
        Arrays.fill(d, Double.MAX_VALUE);

        for (int j = 1; j <= k; j++) {
            final int prev = j - 1;
            final T medoid = medoids[prev];
            IntStream.range(0, n).parallel().forEach(i -> {
                // compute the distance
                double dist = distance.applyAsDouble(data[i], medoid);
                if (dist < d[i]) {
                    d[i] = dist;
                    y[i] = prev;
                }
            });

            if (j < k) {
                double cost = 0.0;
                double cutoff = Calculate.randomInt(n) * Calculate.sum(d);
                for (int index = 0; index < n; index++) {
                    cost += d[index];
                    if (cost >= cutoff) {
                        medoids[j] = data[index];
                        break;
                    }
                }
            }
        }
        return d;
    }

    public String globalInfo() {
        return "Clarans melakukan pencarian cluster terbaik berdasarkan nilai cost minimal atau cost yang paling rendah (mincost)." +
                "Pencarian nilai mincost dilakukan dengan membandingkan node acak current dan neighbor. " +
                "Mincost terbaik dianggap sebagai solusi paling optimal (best node). " +
                "Current akan pindah ke node neighbor jika neighbor adalah pilihan yang lebih baik untuk medoid. " +
                "Jika tidak, berarti local minima (node dengan minimal cost yang terbaik) ditemukan. " +
                "Seluruh proses diulang beberapa kali untuk menemukan yang lebih baik.";
    }

    /**
     * Returns default capabilities of the clusterer.
     *
     * @return the capabilities of this clusterer
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NO_CLASS);

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        return result;
    }

    /**
     * Generates a clusterer. Has to initialize all fields of the clusterer that
     * are not being set via options.
     *
     * @param data set of instances serving as training data
     * @throws Exception if the clusterer has not been generated successfully
     */
    @Override
    public void buildClusterer(Instances data) throws Exception {

        Random r = new Random(getSeed());

        getCapabilities().testWithFail(data);
        ReplaceMissingVal = new ReplaceMissingValues();
        Instances instances = new Instances(data);
        instances.setClassIndex(-1);

        ReplaceMissingVal.setInputFormat(instances);
        instances = Filter.useFilter(instances, ReplaceMissingVal);
    }
    // Returns the number of clusters.
    @Override
    public int numberOfClusters() throws Exception {
        return number_clusters;
    }

    /**
     * Classifies a given instance.
     *
     * @param instance the instance to be assigned to a cluster
     * @return the number of the assigned cluster as an interger if the class is
     *         enumerated, otherwise the predicted value
     * @throws Exception if instance could not be classified successfully
     */
    @Override
    public int clusterInstance(Instance instance) throws Exception {
        return 0;
    }


    /**
     * Sets the OptionHandler's options using the given list.
     * All options will be set (or reset) during this call (i.e. incremental setting of options is not possible).
     *
     * Parameters:
     * options - the list of options as an array of strings
     *
     * Throws:
     * Exception - if an option is not supported
     *
     * Parses a given list of options.
     * <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -NL &lt;integer&gt;
     *  The number of local minima obtained (amount of iterations for solving the problems)
     * </pre>
     *
     * <pre> -MN &lt;integer&gt;
     *  The maximum number of neighbors examined.</pre>
     *
     * <pre> -K &lt;integer&gt;
     *  amount of clusters that should be allocated</pre>
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */

    @Override
    public void setOptions(String[] options) throws Exception {

        String numClusterString = Utils.getOption("clusters", options);
        if (numClusterString.length() > 0) {
            setNumberClusters(Integer.parseInt(numClusterString));
        }

        String numLoc = Utils.getOption("numlocal", options);
        if (numLoc.length() > 0) {
            setNumLocal(Integer.parseInt(numLoc));
        }
        String maxiNeighbor = Utils.getOption("maximumneighbor", options);
        if (maxiNeighbor.length() > 0) {
            setmaximumNeighbor(Integer.parseInt(maxiNeighbor));
        }

        super.setOptions(options);

    }


    @Override
    public String[] getOptions(){
        ArrayList<String> OptionsResult = new ArrayList<String>();

        OptionsResult.add("-clusters");
        OptionsResult.add("" + getNumberClusters());

        OptionsResult.add("-numlocal");
        OptionsResult.add("" + getNumLocal());

        OptionsResult.add("-maximumneighbor");
        OptionsResult.add("" + getmaximumNeighbor());

        Collections.addAll( OptionsResult, super.getOptions());
        return OptionsResult.toArray(new String[1]);

    }


    @Override
    public Enumeration<Option> listOptions(){
        Vector<Option> result = new Vector<Option>();

        result.add(new Option("\tAmount of clusters that should be allocated",
                "clusters", 1, "-clusters <integer>"));

        result.add(new Option("\tThe number of local minima obtained (amount of iterations for solving the problems)",
                "numlocal", 1, "-numlocal <integer>"));

        result.add(new Option("\tThe maximum number of neighbors examined. " +
                "The higher the value of maximumNeighbor, the closer is CLARANS to PAM, and the\n" +
                " * longer is each search of a local minima. But the quality of such a local\n" +
                " * minima is higher and fewer local minima needs to be obtained.",
                "maximumneighbor", 1, "-maximumneighbor <integer>"));

        result.addAll(Collections.list(super.listOptions()));

        return result.elements();
    }

    public static void main(String[] args) {
        runClusterer(new CLARANS(), args);
    }

    private static class Calculate {
        public static double sum(double[] x) {
            double sum = 0.0;
            for (double n : x) {
                sum += n;
            }
            return sum;
        }
        public static int randomInt(int n) {
            for (int i = 0; i < 10; i++) {
                new Random().nextInt();
            }
            return n;
        }
    }


    public List<int[]> getClusters() {return clusters;}
    public List<int[]> getMedoids() {return optimal_medoids;}

    public int getNumberClusters(){ return number_clusters; }
    public void setNumberClusters(int number_clusters){
        this.number_clusters = number_clusters;
    }

    public int getNumLocal(){ return num_local; }
    public void setNumLocal(int num_local){
        this.num_local = num_local;
    }

    public int getmaximumNeighbor(){ return maximumNeighbor; }
    public void setmaximumNeighbor (int maximumNeighbor){
        this.maximumNeighbor = maximumNeighbor;
    }

}
