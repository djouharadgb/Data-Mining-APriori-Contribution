package weka.associations;

import weka.core.*;
import weka.core.Capabilities.Capability;

import java.io.*;
import java.util.*;
import java.util.stream.*;

/**
 * SPEAR: Streaming Positive-nEgative Association Rules.
 * <p>
 * Two key innovations over prior weighted negative association miners:
 * <ol>
 *   <li><b>Entropy-based regularity</b> — uses normalised Shannon entropy
 *       of inter-occurrence gaps instead of fragile max-gap distance.
 *       Threshold adapts per item: reg_score &ge; omega(x).</li>
 *   <li><b>Unified positive + negative rules</b> — discovers both rule types
 *       in a single pass, ranked by a common severity metric.</li>
 * </ol>
 * <p>
 * Weights can be loaded from a properties file (item=weight per line)
 * or fall back to a built-in clinical severity table.
 *
 * @author SPEAR Research Team
 */
public class SPEAR extends AbstractAssociator
        implements OptionHandler, TechnicalInformationHandler {

    private static final long serialVersionUID = 2L;

    // ── Algorithm parameters ─────────────────────────────────
    protected double m_minWSup       = 0.02;
    protected double m_minWNegSup    = 0.01;
    protected double m_minWNegConf   = 0.25;
    protected double m_minPosConf    = 0.50;
    protected double m_defaultWeight = 0.40;
    protected String m_weightsFile   = "";

    // ── Internal state ───────────────────────────────────────
    private Map<String, Double>    m_weightMap = new HashMap<>();
    private List<UnifiedRule>      m_rules     = new ArrayList<>();
    private Map<String, ItemStats> m_allStats  = new HashMap<>();
    private Map<String, ItemStats> m_retained  = new HashMap<>();
    private int  m_numInstances;
    private int  m_prunedWSup;
    private int  m_prunedEntropy;
    private long m_elapsedMs;

    // ══════════════════════════════════════════════════════════
    //  Inner classes
    // ══════════════════════════════════════════════════════════

    private static class ItemStats implements Serializable {
        String name;
        double support;
        double weight;
        double wsup;
        double regScore;   // normalised gap entropy in [0, 1]
        int    maxGap;     // kept for diagnostics
        int    count;
        BitSet tidBits;
    }

    private static class UnifiedRule implements Serializable, Comparable<UnifiedRule> {
        String ruleType;   // "POSITIVE" or "NEGATIVE"
        String antecedent, consequent;
        double supAnt, supCons, supJoint;
        double weightAnt, weightCons;
        double regScoreAnt, regScoreCons;
        // Negative-specific
        double wNegSup, wNegConf;
        // Positive-specific
        double confidence, wconf;
        // Common
        double severity;

        @Override
        public int compareTo(UnifiedRule o) {
            return Double.compare(o.severity, this.severity); // descending
        }
    }

    // ══════════════════════════════════════════════════════════
    //  Capabilities
    // ══════════════════════════════════════════════════════════

    @Override
    public Capabilities getCapabilities() {
        Capabilities caps = super.getCapabilities();
        caps.disableAll();
        caps.enable(Capability.NO_CLASS);
        caps.enable(Capability.NOMINAL_ATTRIBUTES);
        caps.enable(Capability.BINARY_ATTRIBUTES);
        return caps;
    }

    // ══════════════════════════════════════════════════════════
    //  Core algorithm
    // ══════════════════════════════════════════════════════════

    @Override
    public void buildAssociations(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        long t0 = System.currentTimeMillis();

        m_numInstances = data.numInstances();
        m_rules.clear();
        m_allStats.clear();
        m_retained.clear();
        m_prunedWSup    = 0;
        m_prunedEntropy = 0;

        loadWeights();

        int n = data.numInstances();
        int m = data.numAttributes();

        // ── Step 1: per-attribute statistics ──────────────────
        for (int a = 0; a < m; a++) {
            Attribute attr = data.attribute(a);
            String name = attr.name();

            int posIndex = -1;
            if (attr.isNominal()) {
                posIndex = attr.indexOfValue("1");
                if (posIndex < 0) posIndex = attr.numValues() - 1;
            }

            BitSet tidBits = new BitSet(n);
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (!data.instance(i).isMissing(a)) {
                    if (attr.isNominal() && (int) data.instance(i).value(a) == posIndex) {
                        tidBits.set(i);
                        count++;
                    }
                }
            }
            if (count == 0) continue;

            double sup = (double) count / n;
            double w   = getWeight(name);
            double wsup = w * sup;

            // Entropy-based regularity
            double[] regResult = computeEntropyRegularity(tidBits, n);
            double regScore = regResult[0];
            int    maxGap   = (int) regResult[1];

            ItemStats st = new ItemStats();
            st.name     = name;
            st.support  = sup;
            st.weight   = w;
            st.wsup     = wsup;
            st.regScore = regScore;
            st.maxGap   = maxGap;
            st.count    = count;
            st.tidBits  = tidBits;

            m_allStats.put(name, st);
        }

        // ── Process A: item retention ────────────────────────
        for (Map.Entry<String, ItemStats> e : m_allStats.entrySet()) {
            ItemStats s = e.getValue();
            if (s.wsup < m_minWSup) {
                m_prunedWSup++;
                continue;
            }
            // Entropy regularity: reg_score >= omega(x)
            if (s.regScore < s.weight) {
                m_prunedEntropy++;
                continue;
            }
            m_retained.put(e.getKey(), s);
        }

        // ── Process C: unified pairwise rule detection ───────
        List<String> items = new ArrayList<>(m_retained.keySet());
        Collections.sort(items);

        for (int i = 0; i < items.size(); i++) {
            for (int j = i + 1; j < items.size(); j++) {
                String A = items.get(i);
                String B = items.get(j);
                ItemStats sA = m_retained.get(A);
                ItemStats sB = m_retained.get(B);

                BitSet inter = (BitSet) sA.tidBits.clone();
                inter.and(sB.tidBits);
                double supJoint = (double) inter.cardinality() / n;

                // Negative rules (both directions)
                emitNegativeRule(A, B, sA, sB, supJoint);
                emitNegativeRule(B, A, sB, sA, supJoint);

                // Positive rules (both directions)
                if (supJoint > 0) {
                    emitPositiveRule(A, B, sA, sB, supJoint);
                    emitPositiveRule(B, A, sB, sA, supJoint);
                }
            }
        }

        Collections.sort(m_rules);
        m_elapsedMs = System.currentTimeMillis() - t0;
    }

    private void emitNegativeRule(String ant, String cons,
                                  ItemStats sAnt, ItemStats sCons,
                                  double supJoint) {
        double wNegSup = sAnt.weight * sCons.weight * (sAnt.support - supJoint);
        if (wNegSup <= 0 || sAnt.wsup <= 0) return;

        double wNegConf = wNegSup / sAnt.wsup;
        if (wNegSup < m_minWNegSup || wNegConf < m_minWNegConf) return;

        double avgW = (sAnt.weight + sCons.weight) / 2.0;
        double severity = wNegConf * avgW;

        UnifiedRule r = new UnifiedRule();
        r.ruleType     = "NEGATIVE";
        r.antecedent   = ant;
        r.consequent   = cons;
        r.supAnt       = sAnt.support;
        r.supCons      = sCons.support;
        r.supJoint     = supJoint;
        r.weightAnt    = sAnt.weight;
        r.weightCons   = sCons.weight;
        r.regScoreAnt  = sAnt.regScore;
        r.regScoreCons = sCons.regScore;
        r.wNegSup      = wNegSup;
        r.wNegConf     = wNegConf;
        r.severity     = severity;

        m_rules.add(r);
    }

    private void emitPositiveRule(String ant, String cons,
                                  ItemStats sAnt, ItemStats sCons,
                                  double supJoint) {
        double confidence = supJoint / sAnt.support;
        if (confidence < m_minPosConf) return;

        double wconf = confidence * sCons.weight;
        double avgW  = (sAnt.weight + sCons.weight) / 2.0;
        double severity = wconf * avgW;

        UnifiedRule r = new UnifiedRule();
        r.ruleType     = "POSITIVE";
        r.antecedent   = ant;
        r.consequent   = cons;
        r.supAnt       = sAnt.support;
        r.supCons      = sCons.support;
        r.supJoint     = supJoint;
        r.weightAnt    = sAnt.weight;
        r.weightCons   = sCons.weight;
        r.regScoreAnt  = sAnt.regScore;
        r.regScoreCons = sCons.regScore;
        r.confidence   = confidence;
        r.wconf        = wconf;
        r.severity     = severity;

        m_rules.add(r);
    }

    // ── Entropy regularity computation ───────────────────────

    /**
     * Computes normalised gap entropy for an item's occurrence pattern.
     * Returns [regScore, maxGap].
     * regScore in [0, 1]: 1 = perfectly regular, 0 = maximally clustered.
     */
    private double[] computeEntropyRegularity(BitSet tidBits, int n) {
        int card = tidBits.cardinality();
        if (card == 0) return new double[]{0.0, n};

        // Collect gaps (including boundary gaps)
        List<Integer> gaps = new ArrayList<>();
        int prev = -1;
        int maxGap = 0;
        for (int i = tidBits.nextSetBit(0); i >= 0; i = tidBits.nextSetBit(i + 1)) {
            int gap = (prev < 0) ? i : (i - prev);
            gaps.add(gap);
            maxGap = Math.max(maxGap, gap);
            prev = i;
        }
        // Trailing gap
        int trailing = (n - 1) - prev;
        gaps.add(trailing);
        maxGap = Math.max(maxGap, trailing);

        // Shannon entropy of gap distribution
        double total = 0;
        for (int g : gaps) total += g;
        if (total == 0) return new double[]{1.0, 0};  // every position is an occurrence

        int nonZero = 0;
        double entropy = 0.0;
        for (int g : gaps) {
            if (g > 0) {
                double p = g / total;
                entropy -= p * Math.log(p) / Math.log(2);
                nonZero++;
            }
        }

        // Normalise to [0, 1]
        double regScore;
        if (nonZero >= 2) {
            double maxEntropy = Math.log(nonZero) / Math.log(2);
            regScore = (maxEntropy > 0) ? entropy / maxEntropy : 0.0;
        } else {
            regScore = 0.0;
        }

        return new double[]{regScore, maxGap};
    }

    // ── Weight loading ───────────────────────────────────────

    private double getWeight(String itemName) {
        if (m_weightMap.containsKey(itemName)) {
            return m_weightMap.get(itemName);
        }
        String norm = itemName.toUpperCase()
                .replaceAll("[^A-Z0-9]", "_")
                .replaceAll("_+", "_")
                .replaceAll("^_|_$", "");
        if (m_weightMap.containsKey(norm)) {
            return m_weightMap.get(norm);
        }
        return m_defaultWeight;
    }

    private void loadWeights() {
        m_weightMap.clear();
        loadBuiltinWeights();

        if (m_weightsFile != null && !m_weightsFile.isEmpty()) {
            File f = new File(m_weightsFile);
            if (f.exists() && f.canRead()) {
                try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty() || line.startsWith("#")) continue;
                        String[] parts = line.split("=", 2);
                        if (parts.length == 2) {
                            try {
                                m_weightMap.put(parts[0].trim(),
                                        Double.parseDouble(parts[1].trim()));
                            } catch (NumberFormatException ignored) {}
                        }
                    }
                } catch (IOException e) {
                    System.err.println("[SPEAR] Warning: could not read weights file: " + e.getMessage());
                }
            }
        }
    }

    private void loadBuiltinWeights() {
        // Critical (0.90 – 1.00)
        m_weightMap.put("SEIZURES", 1.00);
        m_weightMap.put("HEART_ATTACK", 1.00);
        m_weightMap.put("STROKE", 1.00);
        m_weightMap.put("CARDIAC_ARREST", 1.00);
        m_weightMap.put("LIVER_FAILURE", 0.95);
        m_weightMap.put("KIDNEY_FAILURE", 0.95);
        m_weightMap.put("RESPIRATORY_FAILURE", 0.95);
        m_weightMap.put("ANAPHYLAXIS", 0.95);
        m_weightMap.put("SEVERE_ALLERGIC_REACTION", 0.92);
        m_weightMap.put("SUICIDAL_THOUGHTS", 0.90);
        m_weightMap.put("BLOOD_CLOTS", 0.90);
        // Severe (0.65 – 0.89)
        m_weightMap.put("BLEEDING", 0.85);
        m_weightMap.put("INTERNAL_BLEEDING", 0.88);
        m_weightMap.put("CHEST_PAIN", 0.85);
        m_weightMap.put("DIFFICULTY_BREATHING", 0.85);
        m_weightMap.put("BREATHLESSNESS", 0.85);
        m_weightMap.put("HALLUCINATIONS", 0.80);
        m_weightMap.put("ALLERGIC_REACTION", 0.80);
        m_weightMap.put("IRREGULAR_HEARTBEAT", 0.78);
        m_weightMap.put("INCREASED_LIVER_ENZYMES", 0.78);
        m_weightMap.put("MEMORY_LOSS", 0.78);
        m_weightMap.put("CONFUSION", 0.75);
        m_weightMap.put("ORTHOSTATIC_HYPOTENSION__SUDDEN_LOWERING_OF_BLOOD_PRESSURE_ON_STANDING_", 0.75);
        m_weightMap.put("ORTHOSTATIC_HYPOTENSION", 0.75);
        m_weightMap.put("HIGH_BLOOD_PRESSURE", 0.75);
        m_weightMap.put("LOW_BLOOD_PRESSURE", 0.75);
        m_weightMap.put("TACHYCARDIA", 0.72);
        m_weightMap.put("PALPITATIONS", 0.72);
        m_weightMap.put("INCREASED_HEART_RATE", 0.72);
        m_weightMap.put("ERECTILE_DYSFUNCTION", 0.70);
        m_weightMap.put("VISION_PROBLEMS", 0.70);
        m_weightMap.put("BLURRED_VISION", 0.68);
        m_weightMap.put("SLOW_HEART_RATE", 0.65);
        m_weightMap.put("INCREASED_CREATININE_LEVEL_IN_BLOOD", 0.65);
        m_weightMap.put("DEPRESSION", 0.65);
        // Moderate (0.40 – 0.64)
        m_weightMap.put("VOMITING", 0.60);
        m_weightMap.put("BREAST_ENLARGEMENT_IN_MALE", 0.60);
        m_weightMap.put("SWELLING", 0.60);
        m_weightMap.put("BALANCE_DISORDER__LOSS_OF_BALANCE_", 0.60);
        m_weightMap.put("EDEMA", 0.58);
        m_weightMap.put("FEVER", 0.58);
        m_weightMap.put("ANXIETY", 0.55);
        m_weightMap.put("DIARRHEA", 0.55);
        m_weightMap.put("STOMACH_PAIN", 0.55);
        m_weightMap.put("ABDOMINAL_PAIN", 0.55);
        m_weightMap.put("STOMACH_PAIN_EPIGASTRIC_PAIN", 0.55);
        m_weightMap.put("COLD_EXTREMITIES", 0.55);
        m_weightMap.put("UPPER_RESPIRATORY_TRACT_INFECTION", 0.50);
        m_weightMap.put("FLU_LIKE_SYMPTOMS", 0.50);
        m_weightMap.put("RASH", 0.50);
        m_weightMap.put("GASTROINTESTINAL_DISTURBANCE", 0.50);
        m_weightMap.put("HIVES", 0.52);
        m_weightMap.put("SKIN_RASH", 0.50);
        m_weightMap.put("IMPOTENCE", 0.55);
        m_weightMap.put("ANKLE_SWELLING", 0.55);
        m_weightMap.put("EDEMA__SWELLING_", 0.58);
        m_weightMap.put("UNUSUAL_PRODUCTION_OF_BREAST_MILK_IN_WOMEN_AND_MEN", 0.55);
        m_weightMap.put("SEDATION", 0.55);
        m_weightMap.put("WEAKNESS", 0.45);
        m_weightMap.put("TREMORS", 0.45);
        m_weightMap.put("LEG_CRAMPS", 0.45);
        m_weightMap.put("MUSCLE_CRAMP", 0.45);
        m_weightMap.put("CHILLS", 0.45);
        m_weightMap.put("SLEEPINESS", 0.40);
        m_weightMap.put("FREQUENT_URGE_TO_URINATE", 0.40);
        m_weightMap.put("METALLIC_TASTE", 0.40);
        m_weightMap.put("ENLARGED_SALIVARY_GLAND", 0.40);
        m_weightMap.put("WEIGHT_LOSS", 0.50);
        m_weightMap.put("WEIGHT_GAIN", 0.45);
        // Mild (0.10 – 0.39)
        m_weightMap.put("NAUSEA", 0.35);
        m_weightMap.put("LOSS_OF_APPETITE", 0.35);
        m_weightMap.put("DIZZINESS", 0.35);
        m_weightMap.put("UPSET_STOMACH", 0.35);
        m_weightMap.put("ANAL_IRRITATION", 0.35);
        m_weightMap.put("LIGHTHEADEDNESS", 0.32);
        m_weightMap.put("FATIGUE", 0.32);
        m_weightMap.put("CONSTIPATION", 0.32);
        m_weightMap.put("NERVOUSNESS", 0.32);
        m_weightMap.put("RESTLESSNESS", 0.30);
        m_weightMap.put("HEADACHE", 0.30);
        m_weightMap.put("TIREDNESS", 0.30);
        m_weightMap.put("DROWSINESS", 0.28);
        m_weightMap.put("ITCHING", 0.28);
        m_weightMap.put("HEARTBURN", 0.28);
        m_weightMap.put("INDIGESTION", 0.26);
        m_weightMap.put("DRY_MOUTH", 0.20);
        m_weightMap.put("DRYNESS_IN_MOUTH", 0.20);
        m_weightMap.put("FLUSHING", 0.20);
        m_weightMap.put("FLUSHING__SENSE_OF_WARMTH_IN_THE_FACE__EARS__NECK_AND_TRUNK_", 0.20);
        m_weightMap.put("BLOATING", 0.22);
        m_weightMap.put("SWEATING", 0.24);
        m_weightMap.put("FLATULENCE", 0.18);
        m_weightMap.put("APPLICATION_SITE_REACTIONS__BURNING__IRRITATION__ITCHING_AND_REDNESS_", 0.28);
        m_weightMap.put("INJECTION_SITE_REACTIONS__PAIN__SWELLING__REDNESS_", 0.28);
    }

    // ══════════════════════════════════════════════════════════
    //  Output
    // ══════════════════════════════════════════════════════════

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("=== SPEAR: Streaming Positive-nEgative Association Rules ===\n");
        sb.append("    with Entropy-Based Regularity\n\n");

        sb.append("-- Parameters --\n");
        sb.append(String.format("  minWSup      = %.4f\n", m_minWSup));
        sb.append(String.format("  minWNegSup   = %.4f\n", m_minWNegSup));
        sb.append(String.format("  minWNegConf  = %.4f\n", m_minWNegConf));
        sb.append(String.format("  minPosConf   = %.4f\n", m_minPosConf));
        sb.append(String.format("  defaultWeight= %.2f\n", m_defaultWeight));
        sb.append("  regularity   = entropy (threshold = omega per item)\n");
        if (m_weightsFile != null && !m_weightsFile.isEmpty())
            sb.append("  weightsFile  = ").append(m_weightsFile).append("\n");
        sb.append("\n");

        sb.append("-- Dataset --\n");
        sb.append(String.format("  Instances:   %d\n", m_numInstances));
        sb.append(String.format("  Total items: %d\n", m_allStats.size()));
        sb.append(String.format("  Retained:    %d  (pruned wsup: %d, pruned entropy: %d)\n",
                m_retained.size(), m_prunedWSup, m_prunedEntropy));
        sb.append(String.format("  Time:        %d ms\n\n", m_elapsedMs));

        // Count rule types
        long nNeg = m_rules.stream().filter(r -> r.ruleType.equals("NEGATIVE")).count();
        long nPos = m_rules.stream().filter(r -> r.ruleType.equals("POSITIVE")).count();

        sb.append(String.format("-- Unified Rules: %d  (negative: %d, positive: %d) --\n\n",
                m_rules.size(), nNeg, nPos));

        if (!m_rules.isEmpty()) {
            sb.append(String.format("%-4s %-4s  %-35s    %-30s  %8s %8s %8s %8s %8s\n",
                    "#", "Type", "Antecedent", "Consequent",
                    "Severity", "wNConf", "Conf", "w(A)", "w(B)"));
            sb.append(repeatChar('-', 150)).append("\n");

            int rank = 1;
            for (UnifiedRule r : m_rules) {
                String arrow = r.ruleType.equals("NEGATIVE") ? "=> NOT " : "=>     ";
                double metric = r.ruleType.equals("NEGATIVE") ? r.wNegConf : r.confidence;
                String metricNeg = r.ruleType.equals("NEGATIVE")
                        ? String.format("%8.4f", r.wNegConf) : "     N/A";
                String metricPos = r.ruleType.equals("POSITIVE")
                        ? String.format("%8.4f", r.confidence) : "     N/A";

                sb.append(String.format("%-4d %-4s  %-35s %s%-30s  %8.4f %s %s %8.2f %8.2f\n",
                        rank++,
                        r.ruleType.equals("NEGATIVE") ? "NEG" : "POS",
                        truncate(r.antecedent, 35),
                        arrow,
                        truncate(r.consequent, 30),
                        r.severity, metricNeg, metricPos,
                        r.weightAnt, r.weightCons));
            }
        }

        return sb.toString();
    }

    private static String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max - 2) + "..";
    }

    private static String repeatChar(char c, int n) {
        char[] arr = new char[n];
        Arrays.fill(arr, c);
        return new String(arr);
    }

    // ══════════════════════════════════════════════════════════
    //  WEKA Options Interface
    // ══════════════════════════════════════════════════════════

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> opts = new Vector<>();
        opts.add(new Option("\tMinimum weighted support (default 0.02)", "S", 1, "-S <num>"));
        opts.add(new Option("\tMinimum weighted negative support (default 0.01)", "N", 1, "-N <num>"));
        opts.add(new Option("\tMinimum weighted negative confidence (default 0.25)", "C", 1, "-C <num>"));
        opts.add(new Option("\tMinimum positive confidence (default 0.50)", "P", 1, "-P <num>"));
        opts.add(new Option("\tDefault item weight (default 0.40)", "D", 1, "-D <num>"));
        opts.add(new Option("\tWeights properties file (optional)", "W", 1, "-W <path>"));
        return opts.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String s;
        s = Utils.getOption('S', options);
        if (s.length() > 0) m_minWSup = Double.parseDouble(s);
        s = Utils.getOption('N', options);
        if (s.length() > 0) m_minWNegSup = Double.parseDouble(s);
        s = Utils.getOption('C', options);
        if (s.length() > 0) m_minWNegConf = Double.parseDouble(s);
        s = Utils.getOption('P', options);
        if (s.length() > 0) m_minPosConf = Double.parseDouble(s);
        s = Utils.getOption('D', options);
        if (s.length() > 0) m_defaultWeight = Double.parseDouble(s);
        s = Utils.getOption('W', options);
        if (s.length() > 0) m_weightsFile = s;
    }

    @Override
    public String[] getOptions() {
        List<String> opts = new ArrayList<>();
        opts.add("-S"); opts.add(String.valueOf(m_minWSup));
        opts.add("-N"); opts.add(String.valueOf(m_minWNegSup));
        opts.add("-C"); opts.add(String.valueOf(m_minWNegConf));
        opts.add("-P"); opts.add(String.valueOf(m_minPosConf));
        opts.add("-D"); opts.add(String.valueOf(m_defaultWeight));
        if (m_weightsFile != null && !m_weightsFile.isEmpty()) {
            opts.add("-W"); opts.add(m_weightsFile);
        }
        return opts.toArray(new String[0]);
    }

    // ── WEKA GUI getters/setters ─────────────────────────────

    public double getMinWSup()          { return m_minWSup; }
    public void   setMinWSup(double v)  { m_minWSup = v; }
    public String minWSupTipText()      { return "Minimum weighted support (wsup = omega * support)."; }

    public double getMinWNegSup()          { return m_minWNegSup; }
    public void   setMinWNegSup(double v)  { m_minWNegSup = v; }
    public String minWNegSupTipText()      { return "Minimum weighted negative support for negative rules."; }

    public double getMinWNegConf()          { return m_minWNegConf; }
    public void   setMinWNegConf(double v)  { m_minWNegConf = v; }
    public String minWNegConfTipText()      { return "Minimum weighted negative confidence for negative rules."; }

    public double getMinPosConf()          { return m_minPosConf; }
    public void   setMinPosConf(double v)  { m_minPosConf = v; }
    public String minPosConfTipText()      { return "Minimum confidence for positive rules."; }

    public double getDefaultWeight()          { return m_defaultWeight; }
    public void   setDefaultWeight(double v)  { m_defaultWeight = v; }
    public String defaultWeightTipText()      { return "Default clinical weight for items not in the severity table."; }

    public String getWeightsFile()          { return m_weightsFile; }
    public void   setWeightsFile(String v)  { m_weightsFile = v; }
    public String weightsFileTipText()      { return "Optional .properties file with item=weight lines."; }

    // ══════════════════════════════════════════════════════════
    //  TechnicalInformation
    // ══════════════════════════════════════════════════════════

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation ti = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        ti.setValue(TechnicalInformation.Field.TITLE,
                "SPEAR: Streaming Positive-nEgative Association Rules with Entropy-Based Regularity");
        ti.setValue(TechnicalInformation.Field.YEAR, "2025");
        return ti;
    }

    public String globalInfo() {
        return "SPEAR: Streaming Positive-nEgative Association Rules.\n\n"
                + "Discovers unified positive and negative association rules "
                + "using entropy-based regularity and clinical severity weighting.\n\n"
                + "For more information see:\n"
                + getTechnicalInformation().toString();
    }

    // ══════════════════════════════════════════════════════════
    //  Main — standalone testing
    // ══════════════════════════════════════════════════════════

    public static void main(String[] args) {
        runAssociator(new SPEAR(), args);
    }
}
