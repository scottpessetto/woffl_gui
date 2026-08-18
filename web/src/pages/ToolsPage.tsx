/**
 * Scott's Tools - the secret menu landing page.
 *
 * Renders from GET /api/tools/catalog rather than a hardcoded list, so a tool
 * that has not been ported yet cannot appear here as a dead link. The port
 * from the Streamlit tabs is incremental; this page is honest about it.
 */

import { Link } from "react-router-dom";

import { useToolCatalog } from "../api/hooks";
import { Button, Card, ErrorNote, InfoNote, Section, Spinner } from "../components/ui";
import { setToolsUnlocked } from "../lib/secretMenu";

export default function ToolsPage() {
  const catalog = useToolCatalog(true);

  return (
    <div className="space-y-4">
      <Section
        title="Scott's Tools"
        actions={
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setToolsUnlocked(false)}
            title="Hide the menu again; type the word to bring it back"
          >
            Lock
          </Button>
        }
      >
        <p className="text-sm text-slate-600">You found the secret menu.</p>

        {catalog.isLoading && <Spinner label="Loading tools" />}
        {catalog.isError && <ErrorNote error={catalog.error} />}

        {catalog.data && (
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {catalog.data.tools.map((t) => (
              <Link key={t.id} to={t.path} className="block no-underline">
                <Card className="h-full transition-colors hover:border-blue-300">
                  <div className="font-medium text-slate-800">{t.label}</div>
                  <div className="mt-1 text-xs text-slate-500">{t.caption}</div>
                </Card>
              </Link>
            ))}
          </div>
        )}

        {catalog.data && catalog.data.tools.length === 0 && (
          <InfoNote className="mt-3">No tools are available in this build.</InfoNote>
        )}
      </Section>
    </div>
  );
}
