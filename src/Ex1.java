import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Main class for executing Bayesian Network inference.
 * <p>
 * This class is responsible for:
 * <ul>
 *     <li>Loading the Bayesian network from an XML file (defined in the first line of input.txt).</li>
 *     <li>Running queries listed in the input file using the appropriate inference algorithm.</li>
 * </ul>
 * The results are written to the file {@code output.txt}.
 */
public class Ex1 {

    /**
     * Entry point for the program.
     * Loads the XML file, processes queries from the input file, and writes the results.
     *
     * @param args command line arguments (not used)
     */
    public static void main(String[] args) {
        try {
            // Run queries using Parser logic
            Parser.run("input.txt");

            // Load XML filename from the input file
            String xmlFilename = getXMLFilename("input.txt");

            // Load Bayesian network (not used in main logic here)
            BayesianNetwork bn = Parser.loadXML(xmlFilename);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Extracts the XML filename from the first line of the input file.
     *
     * @param inputFile the path to the input file
     * @return the XML filename (first line in file)
     * @throws Exception if the file cannot be read
     */
    private static String getXMLFilename(String inputFile) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String xmlFile = reader.readLine().trim();
        reader.close();
        return xmlFile;
    }
}
