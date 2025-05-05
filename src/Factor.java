import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents a factor in a Bayesian Network.
 * A factor is a function over a subset of variables mapping assignments to probabilities.
 */
public class Factor {

    // --- Fields ---

    private List<String> variables;
    private Map<List<String>, Double> table;
    private Map<String, Variable> networkVars;
    private static Map<String, String> evidence = new HashMap<>();

    // --- Constructor ---

    /**
     * Constructs a factor over the given variables with access to the network variable map.
     * @param variables the list of variable names in the factor.
     * @param networkVars the global map of all variables in the Bayesian network.
     */
    public Factor(List<String> variables, Map<String, Variable> networkVars) {
        this.variables = new ArrayList<>(variables);
        this.table = new LinkedHashMap<>();
        this.networkVars = networkVars;
    }

    // --- Getters ---

    public List<String> getVariables() {
        return variables;
    }

    public Map<List<String>, Double> getTable() {
        return table;
    }

    public static void setEvidence(Map<String, String> evidenceInput) {
        evidence = new HashMap<>(evidenceInput);
    }

    // --- Factor Construction ---

    /**
     * Constructs a factor from a variable's CPT, considering evidence.
     * @param var the variable to convert to a factor.
     * @param evidenceInput the evidence map.
     * @param netVars the network variables.
     * @return a new Factor object.
     */
    public static Factor fromVariable(Variable var, Map<String, String> evidenceInput, Map<String, Variable> netVars) {
        setEvidence(evidenceInput);
        List<String> factorVars = new ArrayList<>(var.getParents());
        factorVars.add(var.getName());

        Factor factor = new Factor(factorVars, netVars);

        List<List<String>> domains = new ArrayList<>();
        for (String v : factorVars) {
            if (evidence.containsKey(v)) {
                domains.add(Collections.singletonList(evidence.get(v)));
            } else {
                domains.add(netVars.get(v).getOutcomes());
            }
        }

        for (List<String> values : cartesianProduct(domains)) {
            Map<String, String> assignment = new LinkedHashMap<>();
            for (int i = 0; i < factorVars.size(); i++) {
                assignment.put(factorVars.get(i), values.get(i));
            }
            double p = var.getProbability(assignment, netVars);
            if (p != 0.0) {
                factor.table.put(values, p);
            }
        }

        return factor;
    }

    // --- Factor Operations ---

    /**
     * Joins this factor with another factor.
     * @param other the other factor to join with.
     * @param counter optional inference algorithm to track multiplications.
     * @return the joined factor.
     */
    public Factor join(Factor other, InferenceAlgorithm counter) {
        List<String> newVars = new ArrayList<>(this.variables);
        for (String var : other.variables) {
            if (!newVars.contains(var)) newVars.add(var);
        }

        Factor result = new Factor(newVars, networkVars);
        Set<List<String>> seenAssignments = new HashSet<>();

        for (Map.Entry<List<String>, Double> e1 : this.table.entrySet()) {
            for (Map.Entry<List<String>, Double> e2 : other.table.entrySet()) {
                Map<String, String> map1 = toAssignmentMap(this.variables, e1.getKey());
                Map<String, String> map2 = toAssignmentMap(other.variables, e2.getKey());

                if (isCompatible(map1, map2)) {
                    Map<String, String> merged = new LinkedHashMap<>(map1);
                    merged.putAll(map2);

                    List<String> newAssignment = new ArrayList<>();
                    for (String var : newVars) {
                        newAssignment.add(merged.get(var));
                    }

                    if (!seenAssignments.contains(newAssignment)) {
                        seenAssignments.add(newAssignment);
                        double v1 = e1.getValue(), v2 = e2.getValue();
                        if (counter != null) counter.incrementMultiplications();
                        result.getTable().put(newAssignment, v1 * v2);
                    }
                }
            }
        }

        return result;
    }

    /**
     * Eliminates a variable by summing it out.
     * @param var the variable to eliminate.
     * @param counter optional counter for tracking additions.
     * @return a new factor with the variable removed.
     */
    public Factor sumAndRemove(String var, InferenceAlgorithm counter) {
        List<String> newVars = new ArrayList<>(variables);
        int removeIdx = newVars.indexOf(var);
        newVars.remove(removeIdx);

        Factor result = new Factor(newVars, networkVars);
        Map<List<String>, List<Double>> groups = new HashMap<>();

        for (Map.Entry<List<String>, Double> entry : table.entrySet()) {
            List<String> reduced = new ArrayList<>(entry.getKey());
            reduced.remove(removeIdx);
            groups.computeIfAbsent(reduced, k -> new ArrayList<>()).add(entry.getValue());
        }

        for (Map.Entry<List<String>, List<Double>> entry : groups.entrySet()) {
            double sum = entry.getValue().get(0);
            for (int i = 1; i < entry.getValue().size(); i++) {
                sum += entry.getValue().get(i);
                if (counter != null) counter.incrementAdditions();
            }
            result.table.put(entry.getKey(), sum);
        }

        return result;
    }

    /**
     * Restricts a variable in this factor to a given value (used for evidence).
     * @param variable the variable to restrict.
     * @param value the value to restrict to.
     * @return a new factor with the restricted variable removed.
     */
    public Factor restrict(String variable, String value) {
        int varIndex = variables.indexOf(variable);
        List<String> newVars = new ArrayList<>(variables);
        newVars.remove(varIndex);

        Factor restricted = new Factor(newVars, networkVars);
        for (Map.Entry<List<String>, Double> entry : table.entrySet()) {
            List<String> assignment = entry.getKey();
            if (assignment.get(varIndex).equals(value)) {
                List<String> reduced = new ArrayList<>(assignment);
                reduced.remove(varIndex);
                restricted.getTable().put(reduced, entry.getValue());
            }
        }
        return restricted;
    }

    // --- Helper Methods ---

    private static boolean isCompatible(Map<String, String> a1, Map<String, String> a2) {
        for (String key : a1.keySet()) {
            if (a2.containsKey(key) && !a1.get(key).equals(a2.get(key))) return false;
        }
        return true;
    }

    private static Map<String, String> toAssignmentMap(List<String> vars, List<String> values) {
        Map<String, String> map = new LinkedHashMap<>();
        for (int i = 0; i < vars.size(); i++) {
            map.put(vars.get(i), values.get(i));
        }
        return map;
    }

    private static List<List<String>> cartesianProduct(List<List<String>> lists) {
        List<List<String>> result = new ArrayList<>();
        if (lists.isEmpty()) {
            result.add(new ArrayList<>());
            return result;
        }
        List<String> first = lists.get(0);
        List<List<String>> rest = cartesianProduct(lists.subList(1, lists.size()));
        for (String item : first) {
            for (List<String> r : rest) {
                List<String> combo = new ArrayList<>();
                combo.add(item);
                combo.addAll(r);
                result.add(combo);
            }
        }
        return result;
    }

    // --- Utility Checks ---

    /**
     * @return true if the factor contains no entries.
     */
    public boolean isEmpty() {
        return table.isEmpty();
    }

    /**
     * @param evidence a map of evidence variables.
     * @return true if the factor only contains a single row and all variables are assigned in the evidence.
     */
    public boolean isTrivial(Map<String, String> evidence) {
        if (table.size() != 1) return false;
        for (String var : variables) {
            if (!evidence.containsKey(var)) return false;
        }
        return true;
    }
}
