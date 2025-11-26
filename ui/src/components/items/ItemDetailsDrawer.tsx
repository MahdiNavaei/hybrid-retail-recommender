import { useTranslation } from "react-i18next";
import { ItemMetadata } from "../../api/types";

interface Props {
  open: boolean;
  anchorItem: ItemMetadata | null;
  similarItems: ItemMetadata[];
  onClose: () => void;
}

const ItemDetailsDrawer = ({ open, anchorItem, similarItems, onClose }: Props) => {
  const { t } = useTranslation();
  if (!open) return null;

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <div className="item-id">#{anchorItem?.item_id}</div>
            {anchorItem?.category && <div className="item-category">{anchorItem.category}</div>}
          </div>
          <button className="ghost-btn" onClick={onClose}>
            ✕
          </button>
        </div>
        <h3 className="drawer-title">{t("similarItems")}</h3>
        <div className="drawer-grid">
          {similarItems.map((itm) => (
            <div key={itm.item_id} className="card item-card">
              <div className="item-id">#{itm.item_id}</div>
              {itm.category && (
                <div className="item-badge">
                  <span className="badge-label">{t("category")}</span>
                  <span className="badge-value">{itm.category}</span>
                </div>
              )}
              {itm.score !== undefined && (
                <div className="item-score">
                  <span className="badge-label">{t("score")}</span>
                  <span className="badge-value">{itm.score?.toFixed(3)}</span>
                </div>
              )}
              {itm.attributes?.slice(0, 2).map((attr) => (
                <div key={`${itm.item_id}-${attr.name}`} className="item-attribute">
                  <span className="badge-label">{attr.name}</span>
                  <span className="badge-value">{attr.value}</span>
                </div>
              ))}
            </div>
          ))}
          {similarItems.length === 0 && <div className="empty-state">{t("noResults")}</div>}
        </div>
      </div>
    </div>
  );
};

export default ItemDetailsDrawer;
