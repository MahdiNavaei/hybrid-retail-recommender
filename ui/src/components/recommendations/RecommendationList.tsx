import { ItemMetadata } from "../../api/types";
import RecommendationCard from "./RecommendationCard";

interface Props {
  items: ItemMetadata[];
  onViewSimilar: (itemId: string) => void;
  loading?: boolean;
}

const RecommendationList = ({ items, onViewSimilar, loading }: Props) => {
  return (
    <div className={`grid ${loading ? "loading" : ""}`}>
      {items.map((item) => (
        <RecommendationCard key={item.item_id} item={item} onViewSimilar={onViewSimilar} />
      ))}
    </div>
  );
};

export default RecommendationList;
