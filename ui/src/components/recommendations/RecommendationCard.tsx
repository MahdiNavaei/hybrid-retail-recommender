import { useTranslation } from "react-i18next";
import { ItemMetadata } from "../../api/types";

interface Props {
  item: ItemMetadata;
  onViewSimilar: (itemId: string) => void;
}

const RecommendationCard = ({ item, onViewSimilar }: Props) => {
  const { t } = useTranslation();
  return (
    <div className="card item-card">
      <div className="item-id">#{item.item_id}</div>
      {/* EN: Display optional title if available */}
      {/* FA: در صورت وجود عنوان کالا نمایش داده می‌شود */}
      {item.title && <div className="item-title">{item.title}</div>}
      {/* EN: Show category with a label to clarify meaning */}
      {/* FA: دسته کالا همراه با برچسب برای وضوح بیشتر */}
      {item.category && (
        <div className="item-badge">
          <span className="badge-label">{t("category")}</span>
          <span className="badge-value">{item.category}</span>
        </div>
      )}
      {/* EN: Highlight score as the recommendation strength */}
      {/* FA: امتیاز نشان‌دهنده قوت توصیه است */}
      {item.score !== undefined && (
        <div className="item-score">
          <span className="badge-label">{t("score")}</span>
          <span className="badge-value">{item.score?.toFixed(3)}</span>
        </div>
      )}
      {/* EN: Render up to two attributes (property/value) for more item clarity */}
      {/* FA: برای وضوح بیشتر کالا، تا دو ویژگی (property/value) نمایش داده می‌شود */}
      {item.attributes?.slice(0, 2).map((attr) => (
        <div key={`${item.item_id}-${attr.name}`} className="item-attribute">
          <span className="badge-label">{attr.name}</span>
          <span className="badge-value">{attr.value}</span>
        </div>
      ))}
      <button className="ghost-btn" onClick={() => onViewSimilar(item.item_id)}>
        {t("viewSimilar")}
      </button>
    </div>
  );
};

export default RecommendationCard;
