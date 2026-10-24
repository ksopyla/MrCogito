// The palette: drag a layer into a slot or between rows. Dropping a slot or a row back here removes it.
import { PALETTE } from "../engine/defaults";
import { CATEGORY } from "./describe";
import { useDrag, useIsOver } from "./drag";

export function Palette({ onQuickAdd }: { onQuickAdd: (key: string) => void }) {
  const { start, live } = useDrag();
  const over = useIsOver((t) => t.drop === "trash");
  const groups = [...new Set(PALETTE.map((p) => p.group))];
  const removing = live && (live.payload.kind === "cell" || live.payload.kind === "row");
  return (
    <aside className={`palette ${over}`} data-drop="trash">
      {removing ? (
        <div className="trash">Drop here to remove</div>
      ) : (
        <>
          <p className="hint">Drag a layer into a slot (+) or between rows. Click one to add it to the selected row.</p>
          {groups.map((grp) => (
            <div key={grp} className="pal-group">
              <h3>{grp}</h3>
              {PALETTE.filter((p) => p.group === grp).map((p) => (
                <div
                  key={p.key}
                  className={`pal-item cat-${CATEGORY[p.make(512).type]}`}
                  title={p.hint}
                  onPointerDown={(e) => start(e, { kind: "new", key: p.key, label: p.label }, () => onQuickAdd(p.key))}
                >
                  <span className="pal-label">{p.label}</span>
                  <span className="pal-hint">{p.hint}</span>
                </div>
              ))}
            </div>
          ))}
          <div className="pal-group">
            <h3>Wires</h3>
            <p className="hint">Drag from a <span className="dot-inline" /> dot: to a later row for a skip, to a cross-attention or notebook read for keys/values, onto a stack header to start it.</p>
            <h3>Loops</h3>
            <p className="hint">Select rows (shift-click their handles), then "Loop" or "Repeat" in the inspector.</p>
          </div>
        </>
      )}
    </aside>
  );
}
