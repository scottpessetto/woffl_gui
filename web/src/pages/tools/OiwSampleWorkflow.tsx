import { useEffect, useRef, useState } from "react";
import { useOiwSamples } from "../../api/hooks";
import type { OiwSampleComparison, OiwSamplesResponse } from "../../api/types";
import { Button, Card, DataTable, ErrorNote, InfoNote, Metric, Section, Spinner, type Column } from "../../components/ui";
import { downloadCsv } from "../../lib/csv";
import { fmtNum, fmtSigned } from "../../lib/format";

const INPUT = "block mt-1 rounded border border-slate-300 px-2 py-1 text-sm w-full";
const LOG_COLUMNS = ["Date", "Time", "Location", "PPM", "Sampler", "Method", "Notes"].map(key => ({ key, label: key }));
const COLUMNS: Column<OiwSampleComparison>[] = [
  { key: "timestamp", label: "Sample time (Alaska)", render: r => r.timestamp?.slice(0, 19).replace("T", " ") ?? `${r.date} (no time)` },
  { key: "concentration", label: "Lab result", render: r => fmtNum(r.concentration, 2) },
  { key: "oil_pct", label: "Sample oil (%)", render: r => fmtNum(r.oil_pct, 4) },
  { key: "meter_oil_pct", label: "Red Eye oil (%)", render: r => fmtNum(r.meter_oil_pct, 4) },
  { key: "error_pts", label: "Meter - sample (pts)", render: r => fmtSigned(r.error_pts, 4) },
  { key: "flow_bpd", label: "Matched flow (BPD)", render: r => fmtNum(r.flow_bpd) },
  { key: "sample_oil_bopd", label: "Sample rate (BOPD)", render: r => fmtNum(r.sample_oil_bopd, 1) },
  { key: "wc_range_pts", label: "Prior 2-min WC range", render: r => fmtNum(r.wc_range_pts, 2) },
  { key: "status", label: "Pair status", render: r => r.status },
];

type LogRow = Record<string, string>;
const EMPTY: LogRow = { Date: "", Time: "", Location: "V-5317", PPM: "", Sampler: "", Method: "", Notes: "" };
const csvText = (rows: LogRow[]) => [LOG_COLUMNS.map(c => c.key), ...rows.map(r => LOG_COLUMNS.map(c => r[c.key]))]
  .map(row => row.map(v => `"${String(v ?? "").replaceAll('"', '""')}"`).join(",")).join("\r\n");

export function OiwSampleWorkflow({ days }: { days: number }) {
  const [file, setFile] = useState<File | null>(null);
  const [log, setLog] = useState<LogRow[]>([]);
  const [draft, setDraft] = useState<LogRow>({ ...EMPTY });
  const [sheet, setSheet] = useState("OIW Daily");
  const [location, setLocation] = useState("V-5317");
  const [locations, setLocations] = useState<string[]>([]);
  const [units, setUnits] = useState<OiwSamplesResponse["units"]>("unknown");
  const [density, setDensity] = useState("");
  const [rate, setRate] = useState("71000");
  const [rateBasis, setRateBasis] = useState<"liquid" | "water">("liquid");
  const [lag, setLag] = useState("0");
  const [result, setResult] = useState<OiwSamplesResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);
  const sequence = useRef(0);
  const { mutateAsync } = useOiwSamples();

  const invalidate = () => { sequence.current += 1; setResult(null); setError(null); setPending(false); };
  useEffect(() => { invalidate(); }, [days]);
  const edit = (setter: (v: string) => void) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    invalidate(); setter(e.target.value);
  };
  const valid = Number.isFinite(Number(rate)) && Number(rate) >= 1000 && Number(rate) <= 300000
    && lag.trim() !== "" && Number(lag) >= 0 && Number(lag) <= 120
    && (units !== "mg/L" || (Number(density) >= 500 && Number(density) <= 1200));
  const source = file ?? (log.length ? new File([csvText(log)], "field_samples.csv", { type: "text/csv" }) : null);

  async function compare() {
    if (!source || !valid) return;
    const request = ++sequence.current;
    setPending(true); setResult(null); setError(null);
    try {
      const parsed = await mutateAsync({ file: source, location, sheet, units,
        oilDensityKgm3: units === "mg/L" ? Number(density) : undefined,
        waterRateBpd: Number(rate), rateBasis, days, lagMinutes: Number(lag) });
      if (request !== sequence.current) return;
      setResult(parsed); setLocations(parsed.locations_available);
    } catch (err) { if (request === sequence.current) setError(err); }
    finally { if (request === sequence.current) setPending(false); }
  }

  function addSample() {
    invalidate(); setFile(null); setLog(previous => [...previous, { ...draft }]);
    setLocation(draft.Location); setDraft({ ...draft, PPM: "", Notes: "" });
  }

  return <Section title="Validate with field samples">
    <InfoNote>
      Use V-5317 grabs from the Red Eye stream to check the meter. P-5417C is downstream of
      the deoilers and measures a different stream. Capture steady operation and excursions,
      with the sample time, lab method, cleaning history and operating changes in the notes.
      A grab establishes a concentration at that time; it does not establish daily lost barrels.
    </InfoNote>
    <Card className="mt-3">
      <div className="flex flex-wrap gap-3 items-end">
        <label className="text-xs text-slate-600">Import sample log (.csv or .xlsx)
          <input className={INPUT} type="file" accept=".csv,.xlsx" onChange={e => {
            const selected = e.target.files?.[0];
            if (selected) { invalidate(); setFile(selected); setLog([]); setLocations([]); }
            e.target.value = "";
          }} />
        </label>
        <label className="text-xs text-slate-600">Worksheet (XLSX)
          <input className={INPUT} value={sheet} onChange={edit(setSheet)} />
        </label>
        <Button size="sm" variant="ghost" onClick={() => downloadCsv("oiw_sample_template.csv", LOG_COLUMNS, [])}>Download blank template</Button>
      </div>
      <details className="mt-3">
        <summary className="cursor-pointer text-sm text-slate-700">Enter a sample</summary>
        <div className="mt-2 grid gap-2 sm:grid-cols-4">
          {LOG_COLUMNS.map(({ key }) => <label key={key} className="text-xs text-slate-600">
            {key === "PPM" ? "Lab result (units selected below)" : key === "Time" ? "Time (Alaska)" : key}
            <input className={INPUT} type={key === "Date" ? "date" : key === "Time" ? "time" : "text"}
              value={draft[key]} onChange={e => setDraft({ ...draft, [key]: e.target.value })} />
          </label>)}
          <Button size="sm" onClick={addSample} disabled={!draft.Date || !draft.Time || !draft.Location.trim() || draft.PPM.trim() === "" || !Number.isFinite(Number(draft.PPM)) || Number(draft.PPM) < 0}>Add to sample log</Button>
        </div>
        {file && <p className="mt-2 text-xs text-slate-500">Adding a manual sample starts a new log in place of the imported file.</p>}
      </details>
      {log.length > 0 && <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
        <span>{log.length} samples in the manual log</span>
        <Button size="sm" variant="ghost" onClick={() => downloadCsv("oiw_field_samples.csv", LOG_COLUMNS, log)}>Download sample log</Button>
        <Button size="sm" variant="ghost" onClick={() => { invalidate(); setLog([]); }}>Clear log</Button>
      </div>}
      {file && <p className="mt-2 text-xs text-slate-500">Selected: {file.name}</p>}
      <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <label className="text-xs text-slate-600">Sample point
          <input className={INPUT} list="oiw-locations" value={location} onChange={edit(setLocation)} />
          <datalist id="oiw-locations">{locations.map(l => <option key={l} value={l} />)}</datalist>
        </label>
        <label className="text-xs text-slate-600">Lab units
          <select className={INPUT} value={units} onChange={edit(v => setUnits(v as typeof units))}>
            <option value="unknown">Confirm units</option><option value="ppmv">ppm by volume</option><option value="mg/L">mg/L</option>
          </select>
        </label>
        {units === "mg/L" && <label className="text-xs text-slate-600">Oil density (kg/m3)
          <input className={INPUT} type="number" min={500} max={1200} value={density} onChange={edit(setDensity)} placeholder="Lab density" />
        </label>}
        <label className="text-xs text-slate-600">Fallback sample flow (BPD)
          <input className={INPUT} type="number" min={1000} max={300000} value={rate} onChange={edit(setRate)} />
        </label>
        <label className="text-xs text-slate-600">Fallback flow basis
          <select className={INPUT} value={rateBasis} onChange={edit(v => setRateBasis(v as typeof rateBasis))}>
            <option value="liquid">Total liquid</option><option value="water">Water only</option>
          </select>
        </label>
        <label className="text-xs text-slate-600">Meter-to-tap delay (min)
          <input className={INPUT} type="number" min={0} max={120} value={lag} onChange={edit(setLag)} />
        </label>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        One units selection applies to the whole file; import different lab bases separately.
        Enter numeric zero only for a reported zero. Keep below-detection results such as &lt;5 in
        the original log; they are excluded from arithmetic. Paired rates use contemporaneous
        Red Eye stream flow as total liquid. The entered fallback flow is only used for sample-day summaries.
      </p>
      <div className="mt-3 flex items-center gap-3">
        <Button size="sm" onClick={() => void compare()} disabled={!source || !valid || pending}>
          {units === "unknown" ? "Review sample log" : "Compare samples"}
        </Button>
        <span className="text-xs text-slate-500">Comparisons use the selected {days}-day historian window. Download your log before leaving this page.</span>
      </div>
      {!valid && <InfoNote className="mt-2">Enter a flow of 1,000-300,000 BPD, delay of 0-120 minutes and, for mg/L, oil density of 500-1,200 kg/m3.</InfoNote>}
      {pending && <Spinner label="Parsing and matching samples" />}
      {error != null && <ErrorNote error={error} />}
    </Card>
    {result && <Card className="mt-3">
      <div className="flex flex-wrap gap-5">
        <Metric label={`Samples at ${result.location}`} value={String(result.sample_count)} />
        <Metric label="Historian pairs" value={`${result.paired_count} / ${result.sample_count}`} />
        <Metric label="Median meter - sample oil" value={`${fmtSigned(result.median_error_pts, 4)} pts`} sub={`${result.stable_pair_count} pairs without a large recent WC change; diagnostic only`} />
      </div>
      {result.notes.map((note, i) => <p className="mt-2 text-xs text-slate-600" key={i}>{note}</p>)}
      {result.sample_count === 0 && <InfoNote className="mt-2">Choose a sampled location: {result.locations_available.join(", ") || "none"}.</InfoNote>}
      {result.samples.length > 0 && <>
        <div className="my-3"><Button size="sm" variant="ghost" onClick={() => {
          const keys = ["source_row", "date", "timestamp", "location", "concentration", "units", "oil_density_kgm3", "lag_minutes", "comparison_time", "oil_pct", "meter_oil_pct", "error_pts", "flow_bpd", "sample_oil_bopd", "meter_oil_bopd", "wc_age_minutes", "flow_age_minutes", "wc_range_pts", "status", "sampler", "method", "notes"];
          downloadCsv("oiw_sample_comparisons.csv", keys.map(key => ({ key, label: key })),
            result.samples.map(r => ({ ...r, units: result.units, oil_density_kgm3: result.oil_density_kgm3, lag_minutes: result.lag_minutes })));
        }}>Download comparisons</Button></div>
        <DataTable columns={COLUMNS} rows={result.samples} rowKey={r => String(r.source_row)} maxHeight="26rem" sortable />
      </>}
      <details className="mt-3"><summary className="cursor-pointer text-sm">Sample-day concentration summaries</summary>
        <table className="mt-2 w-full text-left text-xs"><thead><tr><th>Date</th><th>Grabs</th><th>Mean ({result.units})</th><th>Rate on fallback flow (BOPD)</th></tr></thead>
          <tbody>{result.daily.map(d => <tr key={d.date}><td>{d.date}</td><td>{d.samples}</td><td>{fmtNum(d.ppm_mean, 2)}</td><td>{fmtNum(d.bopd_mean, 2)}</td></tr>)}</tbody>
        </table>
      </details>
    </Card>}
    <details className="mt-3 text-sm text-slate-700">
      <summary className="cursor-pointer">Use the samples to reduce carryover</summary>
      <ol className="mt-2 list-decimal pl-5 space-y-2">
        <li>Collect paired V-5317 grabs under steady conditions and during meter excursions. Record meter cleaning, lab method and sample transport delay; repeat after cleaning to test for meter bias.</li>
        <li>For low-level or off-setpoint events, review actual interface level and controller/valve response. For events near setpoint, also check level measurement, throughput, residence time and treatment performance. These labels describe indications; they do not prove the cause.</li>
        <li>Within approved operating limits, test one change at a time with operations. Compare upstream samples and event duration at similar flow; take downstream samples after the appropriate transit time to check whether final residual oil also improves.</li>
        <li>Repeat the comparison on later samples. Record the change and result in the log; a few selected grabs cannot establish daily recovered oil or a permanent meter correction.</li>
      </ol>
    </details>
  </Section>;
}
