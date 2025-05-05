import java.io.*;
import java.util.*;
import javax.xml.parsers.*;
import org.w3c.dom.*;

/**
 * Parser for reading a Bayesian Network from an XML file,
 * and running inference queries from a text file.
 * Supports algorithms:
 * - 1: Simple Inference
 * - 2: Variable Elimination
 * - 3: Heuristic Variable Elimination
 */
public class Parser {

    // --- XML Parsing ---

    /**
     * Loads a Bayesian Network from an XML file in BIF format.
     * @param filename path to the XML file.
     * @return a fully constructed BayesianNetwork object.
     * @throws Exception if XML parsing fails.
     */
    public static BayesianNetwork loadXML(String filename) throws Exception {
        BayesianNetwork bn = new BayesianNetwork();

        File xmlFile = new File(filename);
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(xmlFile);
        doc.getDocumentElement().normalize();

        // Parse variables and their outcomes
        NodeList varNodes = doc.getElementsByTagName("VARIABLE");
        for (int i = 0; i < varNodes.getLength(); i++) {
            Element varElem = (Element) varNodes.item(i);
            String name = varElem.getElementsByTagName("NAME").item(0).getTextContent();
            Variable var = new Variable(name);

            NodeList outcomeNodes = varElem.getElementsByTagName("OUTCOME");
            for (int j = 0; j < outcomeNodes.getLength(); j++) {
                var.addOutcome(outcomeNodes.item(j).getTextContent());
            }

            bn.addVariable(var);
        }

        // Parse definitions: parent structure and CPTs
        NodeList defNodes = doc.getElementsByTagName("DEFINITION");
        for (int i = 0; i < defNodes.getLength(); i++) {
            Element defElem = (Element) defNodes.item(i);
            String forVar = defElem.getElementsByTagName("FOR").item(0).getTextContent();
            Variable var = bn.getVariable(forVar);

            NodeList children = defElem.getChildNodes();
            for (int j = 0; j < children.getLength(); j++) {
                Node node = children.item(j);
                if (node.getNodeType() == Node.ELEMENT_NODE && node.getNodeName().equals("GIVEN")) {
                    var.addParent(node.getTextContent().trim());
                }
            }

            String tableStr = defElem.getElementsByTagName("TABLE").item(0).getTextContent().trim();
            String[] probs = tableStr.split("\\s+");
            double[] cpt = new double[probs.length];
            for (int j = 0; j < probs.length; j++) {
                cpt[j] = Double.parseDouble(probs[j]);
            }

            var.setCPT(cpt);
        }

        return bn;
    }

    // --- Running Queries ---

    /**
     * Runs inference queries from a text file.
     * First line should contain the XML file path.
     * Following lines are queries, each in the format:
     *   - Joint: P(A=T,B=F)
     *   - Conditional: P(A=T | B=F,C=F),<algorithm>
     * @param inputFile the path to the input queries file.
     * @throws Exception on file read or parsing error.
     */
    public static void run(String inputFile) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String xmlFile = reader.readLine().trim();
        BayesianNetwork bn = loadXML(xmlFile);

        List<String> outputs = new ArrayList<>();
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            if (line.contains("|")) {
                // Conditional query
                try {
                    String queryPart, evidencePart;
                    int algo = 1;

                    // Extract algorithm number (e.g., ,1)
                    if (line.contains(",")) {
                        int commaIndex = line.lastIndexOf(",");
                        String before = line.substring(0, commaIndex).trim();
                        String after = line.substring(commaIndex + 1).trim();
                        algo = Integer.parseInt(after);
                        line = before;
                    }

                    String[] parts = line.split("\\|");
                    queryPart = parts[0].trim();
                    evidencePart = parts[1].trim();

                    // Remove P(...) wrapping
                    if (queryPart.startsWith("P(")) queryPart = queryPart.substring(2);
                    if (queryPart.endsWith(")")) queryPart = queryPart.substring(0, queryPart.length() - 1);

                    String[] queryAssignments = queryPart.split(",");
                    String[] queryVarVal = queryAssignments[0].split("=");
                    String queryVar = queryVarVal[0].trim();
                    String queryValue = queryVarVal[1].trim();

                    Map<String, String> evidence = parseAssignment(evidencePart);

                    double result;
                    int mult, add;

                    // Select and run inference algorithm
                    if (algo == 1) {
                        SimpleInference inf = new SimpleInference(bn);
                        result = inf.query(queryVar, queryValue, evidence);
                        mult = inf.getMultiplicationCount();
                        add = inf.getAdditionCount();
                    } else if (algo == 2) {
                        VariableElimination inf = new VariableElimination(bn);
                        result = inf.query(queryVar, queryValue, evidence);
                        mult = inf.getMultiplicationCount();
                        add = inf.getAdditionCount();
                    } else if (algo == 3) {
                        HeuristicVariableElimination inf = new HeuristicVariableElimination(bn);
                        result = inf.query(queryVar, queryValue, evidence);
                        mult = inf.getMultiplicationCount();
                        add = inf.getAdditionCount();
                    } else {
                        System.err.println("Invalid algorithm number: " + algo);
                        continue;
                    }

                    outputs.add(String.format("%.5f,%d,%d", result, add, mult));
                } catch (Exception e) {
                    System.err.println("Failed to parse conditional query: " + line);
                    e.printStackTrace();
                }
            }

            else if (line.startsWith("P(")) {
                // Joint probability query
                try {
                    Map<String, String> fullAssign = parseAssignment(line);
                    double prob = 1.0;
                    int multCount = 0;

                    for (Variable var : bn.getVariablesInTopologicalOrder()) {
                        double p = var.getProbability(fullAssign, bn.getVariableMap());
                        prob *= p;
                        multCount++;
                    }

                    outputs.add(String.format("%.5f,0,%d", prob, multCount - 1));
                } catch (Exception e) {
                    System.err.println("Failed to parse joint query: " + line);
                    e.printStackTrace();
                }
            }

            else {
                System.err.println("Unrecognized query format: " + line);
            }
        }

        reader.close();

        BufferedWriter writer = new BufferedWriter(new FileWriter("output.txt"));
        for (String out : outputs) {
            writer.write(out);
            writer.newLine();
        }
        writer.close();
    }

    // --- Assignment Parsing Helper ---

    /**
     * Parses an assignment string into a variable-value map.
     * Example: "P(A=T,B=F)" → {A=T, B=F}
     * @param assignmentStr the string to parse.
     * @return map of variable names to values.
     */
    private static Map<String, String> parseAssignment(String assignmentStr) {
        Map<String, String> map = new HashMap<>();
        assignmentStr = assignmentStr.replace("P(", "").replace(")", "").trim();
        String[] pairs = assignmentStr.split(",");
        for (String pair : pairs) {
            String[] keyVal = pair.split("=");
            if (keyVal.length == 2) {
                map.put(keyVal[0].trim(), keyVal[1].trim());
            }
        }
        return map;
    }
}
