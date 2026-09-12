import { Link } from "react-router-dom";

/** Real wells can open their historical prediction evidence from a run row. */
export function WellHistoryLink({ well }: { well: string }) {
  if (!/^MP[A-Z]-\d{1,3}$/.test(well)) return <>{well}</>;
  return <Link className="text-blue-700 underline decoration-blue-200 underline-offset-2 hover:decoration-blue-700"
    title={`Review ${well} production and model match history`}
    to={`/jp-history?well=${encodeURIComponent(well)}&match=1`}>{well}</Link>;
}
