import React from "react";

function getRiskColor(score) {
  if (score >= 80) return "#22c55e";
  if (score >= 50) return "#f59e0b";
  if (score >= 20) return "#f97316";
  return "#ef4444";
}

export default function RiskGauge({ risk }) {
  const score =
    typeof risk?.score === "number"
      ? risk.score
      : typeof risk?.liquidity_score === "number"
      ? risk.liquidity_score
      : null;

  const level = risk?.level || risk?.risk_level || "—";

  if (score === null) return null;

  const color = getRiskColor(score);

  return (
    <div className="kpiCard">
      <div className="kpiLabel">Liquidity Risk</div>

      <div className="kpiValue" style={{ color }}>
        {score}
      </div>

      <div className="kpiSub" style={{ color }}>
        {level}
      </div>
    </div>
  );
}