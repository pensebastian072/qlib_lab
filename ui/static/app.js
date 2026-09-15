// Shared display helpers. Read-only plumbing; there are no actions to take.
const $ = (id) => document.getElementById(id);
async function getJSON(u){ const r = await fetch(u); return r.json(); }
const fmt = (n,d=2) => (n==null||isNaN(n))?'–':Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
const esc = (s) => (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// Header pills: which data source is being rendered, and its as-of date.
// "snapshot" is the normal state on a fresh clone and is not a warning.
function setSourcePills(d){
  const src = d._source || 'none';
  const label = {live:'live flag', snapshot:'shipped snapshot', none:'no data'}[src] || src;
  const cls = {live:'bg-blue', snapshot:'bg-secondary', none:'bg-red'}[src] || 'bg-secondary';
  const pill = $('pillSource');
  if (pill){
    pill.className = 'badge ' + cls;
    pill.textContent = label;
    pill.title = d._source_path || '';
  }
  const asOf = $('pillAsOf');
  if (asOf){
    asOf.textContent = 'as of ' + String(d.as_of || '?').replace('T',' ').slice(0,16);
    asOf.className = 'badge ' + (d.stale ? 'bg-orange' : 'bg-secondary');
  }
}
