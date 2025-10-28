import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const App = () => {
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictionsHistory, setPredictionsHistory] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // --- Load model_data.json from /public ---
  useEffect(() => {
    const loadModelData = async () => {
      try {
        const response = await fetch("/model_data.json");
        const text = await response.text();
        try {
          const data = JSON.parse(text);
          setModelData(data);
        } catch {
          throw new Error(
            "Invalid JSON. Ensure model_data.json is in the /public folder."
          );
        }
      } catch (err) {
        console.error("Error loading model data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadModelData();
  }, []);

  // --- KNN Prediction Logic ---
  const predictLoan = useCallback(
    (input) => {
      if (!modelData) return null;

      const { normalization_ranges, training_data_initial } = modelData;

      const normalize = (val, key) => {
        const { min, max } = normalization_ranges[key];
        return (val - min) / (max - min);
      };

      const normalizedInput = [
        normalize(input.dependents, "dependents"),
        input.education,
        normalize(input.income, "income"),
        normalize(input.loan_amount, "loan_amount"),
        normalize(input.cibil, "cibil"),
        normalize(input.assets_total, "assets_total"),
      ];

      // Euclidean Distance
      const distances = training_data_initial.map((record) => {
        const features = record.slice(0, 6);
        const label = record[6];
        const distance = Math.sqrt(
          features.reduce(
            (sum, val, i) => sum + Math.pow(val - normalizedInput[i], 2),
            0
          )
        );
        return { distance, label };
      });

      // KNN (k=5)
      const k = 5;
      const nearest = distances.sort((a, b) => a.distance - b.distance).slice(0, k);
      const approvedCount = nearest.filter((n) => n.label === 1).length;

      return approvedCount > k / 2 ? "Approved" : "Rejected";
    },
    [modelData]
  );

  // --- Form Data ---
  const [formData, setFormData] = useState({
    dependents: 2,
    education: "Graduate",
    income: 600000,
    loan_amount: 200000,
    cibil: 700,
    assets_total: 500000,
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === "education" ? value : Number(value),
    }));
  };

  const handlePredict = () => {
    const lakhInput = {
      ...formData,
      education: formData.education === "Graduate" ? 1 : 0,
      income: formData.income / 100000,
      loan_amount: formData.loan_amount / 100000,
      assets_total: formData.assets_total / 100000,
    };

    const prediction = predictLoan(lakhInput);
    if (prediction) {
      const newRecord = {
        id: predictionsHistory.length + 1,
        timestamp: new Date().toLocaleTimeString(),
        ...formData,
        prediction,
      };
      setPredictionsHistory((prev) => [newRecord, ...prev]);
    }
  };

  // --- Chart Data ---
  const chartData = useMemo(
    () =>
      predictionsHistory.map((p) => ({
        name: p.timestamp,
        CIBIL: p.cibil,
        "Loan Amount": p.loan_amount,
      })),
    [predictionsHistory]
  );

  const approvalTrendData = useMemo(
    () =>
      predictionsHistory.map((p) => ({
        name: p.timestamp,
        Approved: p.prediction === "Approved" ? 1 : 0,
        Rejected: p.prediction === "Rejected" ? 1 : 0,
      })),
    [predictionsHistory]
  );

  const pieData = useMemo(() => {
    const approved = predictionsHistory.filter((p) => p.prediction === "Approved").length;
    const rejected = predictionsHistory.filter((p) => p.prediction === "Rejected").length;
    return [
      { name: "Approved", value: approved },
      { name: "Rejected", value: rejected },
    ];
  }, [predictionsHistory]);

  const COLORS = ["#22C55E", "#EF4444"];

  if (loading) return <div className="p-6 text-center">Loading model data...</div>;
  if (error) return <div className="p-6 text-red-500">Error: {error}</div>;
  if (!modelData)
    return <div className="p-6 text-center">Model data not available.</div>;

  return (
    <div
      className={`min-h-screen w-full overflow-x-hidden transition-colors duration-300 ${
        isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"
      }`}
    >
      {/* --- Full-Width Container --- */}
      <div className="w-full max-w-[1800px] mx-auto py-10 px-8">
        {/* --- Header --- */}
        <header className="flex justify-between items-center mb-10">
          <h1 className="text-3xl font-extrabold text-indigo-500">
            Loan Approval Prediction Dashboard
          </h1>
          <button
            onClick={() => setIsDarkMode((p) => !p)}
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg shadow hover:bg-indigo-700 transition"
          >
            {isDarkMode ? "Light Mode" : "Dark Mode"}
          </button>
        </header>

        {/* --- Input Form --- */}
        <div
          className={`rounded-xl shadow-lg p-6 ${
            isDarkMode ? "bg-gray-800" : "bg-white"
          }`}
        >
          <h2 className="text-xl font-semibold mb-6 text-indigo-400">
            Input Parameters
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6">
            {Object.keys(formData).map((key) => (
              <div key={key}>
                <label className="block mb-1 text-sm font-medium capitalize">
                  {key.replace("_", " ")}{" "}
                  {["income", "loan_amount", "assets_total"].includes(key)
                    ? "(₹)"
                    : ""}
                </label>

                {key === "education" ? (
                  <select
                    name="education"
                    value={formData.education}
                    onChange={handleChange}
                    className={`w-full px-3 py-2 rounded-md border text-base ${
                      isDarkMode
                        ? "bg-gray-700 border-gray-600 text-white"
                        : "border-gray-300 text-black"
                    }`}
                  >
                    <option value="Graduate">Graduate</option>
                    <option value="Not Graduate">Not Graduate</option>
                  </select>
                ) : (
                  <input
                    type="number"
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                    placeholder={
                      ["income", "loan_amount", "assets_total"].includes(key)
                        ? "in Rupees"
                        : ""
                    }
                    className={`w-full px-3 py-2 rounded-md border text-base ${
                      isDarkMode
                        ? "bg-gray-700 border-gray-600 text-white"
                        : "border-gray-300 text-black"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>

          <button
            onClick={handlePredict}
            className="mt-6 bg-green-600 text-white px-5 py-2 rounded-lg shadow hover:bg-green-700 transition"
          >
            Predict Loan Status
          </button>

          {modelData.initial_accuracy && (
            <p className="mt-4 text-sm text-gray-400">
              Model Accuracy: {modelData.initial_accuracy}%
            </p>
          )}
        </div>

        {/* --- Prediction History & Charts --- */}
        {predictionsHistory.length > 0 && (
          <div className="mt-12 space-y-10">
            {/* --- History Table --- */}
            <div
              className={`rounded-xl shadow-lg p-6 ${
                isDarkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              <h2 className="text-xl font-semibold mb-4 text-indigo-400">
                Prediction History
              </h2>

              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead>
                    <tr>
                      <th className="px-4 py-2">Time</th>
                      <th className="px-4 py-2">CIBIL</th>
                      <th className="px-4 py-2">Loan (₹)</th>
                      <th className="px-4 py-2">Education</th>
                      <th className="px-4 py-2">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictionsHistory.map((item) => (
                      <tr key={item.id}>
                        <td className="px-4 py-2">{item.timestamp}</td>
                        <td className="px-4 py-2">{item.cibil}</td>
                        <td className="px-4 py-2">
                          ₹{item.loan_amount.toLocaleString("en-IN")}
                        </td>
                        <td className="px-4 py-2">{item.education}</td>
                        <td
                          className={`px-4 py-2 font-semibold ${
                            item.prediction === "Approved"
                              ? "text-green-500"
                              : "text-red-500"
                          }`}
                        >
                          {item.prediction}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* --- Charts --- */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
              {/* CIBIL vs Loan Chart */}
              <div
                className={`rounded-xl shadow-lg p-6 ${
                  isDarkMode ? "bg-gray-800" : "bg-white"
                }`}
              >
                <h3 className="text-lg mb-2 font-semibold text-indigo-400">
                  CIBIL vs Loan Amount
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="CIBIL"
                      stroke="#6366F1"
                      strokeWidth={2}
                      activeDot={{ r: 6 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="Loan Amount"
                      stroke="#10B981"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Approval Trend Chart */}
              <div
                className={`rounded-xl shadow-lg p-6 ${
                  isDarkMode ? "bg-gray-800" : "bg-white"
                }`}
              >
                <h3 className="text-lg mb-2 font-semibold text-indigo-400">
                  Approval Trend (Approved vs Rejected)
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={approvalTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="Approved"
                      stroke="#22C55E"
                      strokeWidth={3}
                      dot={{ r: 5 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="Rejected"
                      stroke="#EF4444"
                      strokeWidth={3}
                      dot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Pie Chart Summary */}
            <div
              className={`rounded-xl shadow-lg p-6 flex flex-col items-center ${
                isDarkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              <h3 className="text-lg mb-4 font-semibold text-indigo-400">
                Overall Approval Ratio
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
