import { useEffect, useState } from "react";
import { api } from "./api";
import "./styles.css";

type View =
  | "overview" | "datasets" | "training" | "prediction"
  | "explain" | "benchmark" | "advantage" | "reports";

const NAV: [View, string][] = [
  ["overview", "Overview"],
  ["datasets", "Datasets"],
  ["training", "Training"],
  ["prediction", "Prediction"],
  ["explain", "Explainability"],
  ["benchmark", "Benchmark"],
  ["advantage", "Quantum Advantage"],
  ["reports", "Research Reports"],
];

function useFetch<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    fn().then((d) => live && setData(d))
      .catch((e) => live && setError(String(e)))
      .finally(() => live && setLoading(false));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading };
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="card"><h3 style={{ marginTop: 0 }}>{title}</h3>{children}</section>;
}

function MetricBars({ models }: { models: Record<string, { mean: number; ci_lower: number; ci_upper: number }> }) {
  return (
    <table>
      <thead><tr><th>Model</th><th>AUROC</th><th>95% CI</th><th></th></tr></thead>
      <tbody>
        {Object.entries(models).map(([m, v]) => {
          const q = ["vqc", "qsvm", "reupload", "hybrid_qnn"].includes(m);
          return (
            <tr key={m}>
              <td><span className={`badge ${q ? "quantum" : "classical"}`}>{m}</span></td>
              <td className="mono">{v.mean.toFixed(4)}</td>
              <td className="mono dim">[{v.ci_lower.toFixed(3)}, {v.ci_upper.toFixed(3)}]</td>
              <td style={{ minWidth: 120 }}>
                <div className={`bar ${q ? "quantum" : ""}`}><div style={{ width: `${v.mean * 100}%` }} /></div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Overview() {
  const h = useFetch(api.health);
  const ds = useFetch(api.datasets);
  const ex = useFetch(api.experiments);
  if (h.loading || ds.loading || ex.loading) return <p className="dim">Loading…</p>;
  if (h.error) return <p className="error">Backend unavailable: {h.error}</p>;
  return (
    <>
      <Card title="Project status">
        <p>Backend: <span className="badge ok">{(h.data as any).status}</span>{" "}
          <span className="dim">{(h.data as any).safety}</span></p>
        <p className="dim">Datasets: {(ds.data as any)?.datasets?.length ?? 0} ·
          Experiments: {(ex.data as any)?.experiments?.length ?? 0}</p>
      </Card>
      <Card title="Datasets">
        <div className="grid">
          {(ds.data as any)?.datasets?.map((d: any) => (
            <div key={d.name} className="card" style={{ margin: 0 }}>
              <strong>{d.name}</strong><br />
              <span className="dim mono">{d.samples ?? "?"} samples · {d.features ?? "?"} features ·
                pos {d.pos_rate ?? "?"}</span>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}

function Datasets() {
  const { data, error, loading } = useFetch(api.datasets);
  if (loading) return <p className="dim">Loading…</p>;
  if (error) return <p className="error">{error}</p>;
  return (
    <Card title="Registered datasets (live from backend)">
      <table><thead><tr><th>Name</th><th>Samples</th><th>Features</th><th>Pos rate</th><th>Status</th></tr></thead>
        <tbody>{(data as any).datasets.map((d: any) => (
          <tr key={d.name}><td>{d.name}</td><td className="mono">{d.samples ?? "—"}</td>
            <td className="mono">{d.features ?? "—"}</td><td className="mono">{d.pos_rate ?? "—"}</td>
            <td>{d.error ? <span className="badge bad">error</span> : <span className="badge ok">valid</span>}</td>
          </tr>))}</tbody></table>
    </Card>
  );
}

function Training() {
  const models = useFetch(api.models);
  const [dataset, setDataset] = useState("breast_cancer");
  const [model, setModel] = useState("logistic_regression");
  const [folds, setFolds] = useState(2);
  const [job, setJob] = useState<any>(null);
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    if (!job) return;
    const t = setInterval(async () => {
      const s = await api.experiment(job.experiment_id);
      setStatus(s);
      if (s.status === "completed" || s.status === "failed") clearInterval(t);
    }, 2000);
    return () => clearInterval(t);
  }, [job]);

  return (
    <Card title="Launch benchmark job">
      <label>Dataset
        <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
          {["breast_cancer", "heart_disease", "parkinsons", "diabetes"].map((d) => (
            <option key={d} value={d}>{d}</option>))}</select></label>
      <label>Primary model
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {(models.data as any)?.models?.map((m: string) => (
            <option key={m} value={m}>{m}</option>)) ?? []}</select></label>
      <label>Outer folds
        <input type="number" value={folds} min={2} max={5}
          onChange={(e) => setFolds(Number(e.target.value))} /></label>
      <button className="primary" onClick={async () => {
        setStatus({ status: "queued" });
        const j = await api.train(
          { dataset, model: { type: model }, training: { outer_folds: folds, seeds: [42] } },
          [model]);
        setJob(j);
      }}>Start training</button>
      {status && (
        <p>Job <span className="mono">{job?.experiment_id}</span>:{" "}
          <span className="badge eval status">{status.status}</span>
          {status.results && <> · AUROC {Object.entries(status.results.models as object)
            .map(([m, v]: any) => `${m}=${v.mean.toFixed(3)}`).join(", ")}</>}</p>
      )}
    </Card>
  );
}

function Prediction() {
  const [expId, setExpId] = useState("");
  const [features, setFeatures] = useState("");
  const [out, setOut] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  return (
    <Card title="Predict (fitted experiment model)">
      <label>Experiment ID <input value={expId} onChange={(e) => setExpId(e.target.value)}
        placeholder="exp_xxxxxxxx" /></label>
      <label>Features (comma-separated, original order)
        <textarea rows={3} value={features} onChange={(e) => setFeatures(e.target.value)}
          placeholder="5.1, 3.5, 1.4, ..." /></label>
      <button className="primary" onClick={async () => {
        setError(null);
        try {
          const r = await api.predict(expId, features.split(",").map(Number));
          setOut(r);
        } catch (e) { setError(String(e)); }
      }}>Predict</button>
      {error && <p className="error">{error}</p>}
      {out && <p>Prediction: <strong>{out.prediction}</strong> · p={out.probability.toFixed(4)}{" "}
        {out.abstained
          ? <span className="badge bad">abstained (uncertain)</span>
          : <span className="badge ok">prediction available</span>}</p>}
    </Card>
  );
}

function Explain() {
  const [expId, setExpId] = useState("");
  const [features, setFeatures] = useState("");
  const [out, setOut] = useState<any>(null);
  return (
    <Card title="Local sensitivity explanation (real model)">
      <label>Experiment ID <input value={expId} onChange={(e) => setExpId(e.target.value)} /></label>
      <label>Features (comma-separated)
        <textarea rows={3} value={features} onChange={(e) => setFeatures(e.target.value)} /></label>
      <button className="primary" onClick={async () => {
        setOut(await api.explain(expId, features.split(",").map(Number)));
      }}>Explain</button>
      {out && (
        <table><thead><tr><th>Feature</th><th>d p / d x</th><th></th></tr></thead>
          <tbody>{out.sensitivity.slice(0, 15).map((s: any) => (
            <tr key={s.feature}><td>{s.feature}</td>
              <td className="mono">{s.sensitivity.toFixed(4)}</td>
              <td style={{ minWidth: 100 }}>
                <div className="bar"><div style={{
                  width: `${Math.min(100, Math.abs(s.sensitivity) * 50)}%` }} /></div>
              </td></tr>))}</tbody></table>)}
    </Card>
  );
}

function Benchmark() {
  const [expId, setExpId] = useState("");
  const [out, setOut] = useState<any>(null);
  return (
    <>
      <Card title="Benchmark results">
        <label>Experiment ID <input value={expId} onChange={(e) => setExpId(e.target.value)} /></label>
        <button className="primary" onClick={async () => setOut(await api.benchmark(expId))}>Load</button>
      </Card>
      {out && <Card title={`${out.dataset} · n=${out.n_train}`}>
        <MetricBars models={out.models} />
        <p className="dim">Honesty gap (quantum − classical on same compressed features):{" "}
          <span className="mono">{out.honesty_gap?.toFixed?.(4)}</span></p>
      </Card>}
    </>
  );
}

function Advantage() {
  const [expId, setExpId] = useState("");
  const [out, setOut] = useState<any>(null);
  return (
    <>
      <Card title="Quantum Advantage Certificate">
        <label>Experiment ID <input value={expId} onChange={(e) => setExpId(e.target.value)} /></label>
        <button className="primary" onClick={async () => setOut(await api.certificate(expId))}>Load</button>
      </Card>
      {out && <Card title={`Verdict: ${out.evidence_classification}`}>
        <table><tbody>
          <tr><td>Quantum AUROC</td><td className="mono">{out.quantum.mean.toFixed(4)} [{out.quantum.ci_lower.toFixed(3)}, {out.quantum.ci_upper.toFixed(3)}]</td></tr>
          <tr><td>Best classical AUROC</td><td className="mono">{out.classical_best.mean.toFixed(4)} [{out.classical_best.ci_lower.toFixed(3)}, {out.classical_best.ci_upper.toFixed(3)}]</td></tr>
          <tr><td>Config hash</td><td className="mono">{out.configuration_hash}</td></tr>
          <tr><td>Seeds</td><td className="mono">{out.seeds.join(", ")}</td></tr>
        </tbody></table>
        <p className="dim">{out.safety_note}</p>
      </Card>}
    </>
  );
}

function Reports() {
  const { data, loading, error } = useFetch(api.experiments);
  if (loading) return <p className="dim">Loading…</p>;
  if (error) return <p className="error">{error}</p>;
  return (
    <Card title="Experiments">
      <table><thead><tr><th>ID</th><th>Status</th><th>Detail</th></tr></thead>
        <tbody>{(data as any).experiments.map((e: any) => (
          <tr key={e.id}><td className="mono">{e.id}</td>
            <td><span className="badge eval status">{e.status}</span></td>
            <td className="dim mono">{e.elapsed_s ? `${e.elapsed_s}s` : e.error ?? ""}</td></tr>))}</tbody></table>
    </Card>
  );
}

export default function App() {
  const [view, setView] = useState<View>("overview");
  return (
    <div className="app">
      <nav className="sidebar" aria-label="Primary">
        <h1>QED-Net 2.0</h1>
        <p className="safety">Research use only. Not a medical device; not for diagnosis.</p>
        {NAV.map(([v, label]) => (
          <button key={v} aria-current={view === v ? "page" : undefined}
            onClick={() => setView(v)}>{label}</button>
        ))}
      </nav>
      <main>
        {view === "overview" && <><h2>Overview</h2><Overview /></>}
        {view === "datasets" && <><h2>Datasets</h2><Datasets /></>}
        {view === "training" && <><h2>Training</h2><Training /></>}
        {view === "prediction" && <><h2>Prediction</h2><Prediction /></>}
        {view === "explain" && <><h2>Explainability</h2><Explain /></>}
        {view === "benchmark" && <><h2>Benchmark</h2><Benchmark /></>}
        {view === "advantage" && <><h2>Quantum Advantage</h2><Advantage /></>}
        {view === "reports" && <><h2>Research Reports</h2><Reports /></>}
      </main>
    </div>
  );
}
