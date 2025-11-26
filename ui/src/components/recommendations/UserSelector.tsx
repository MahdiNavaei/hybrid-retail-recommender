import { useState } from "react";
import { useTranslation } from "react-i18next";
import { RecommendationRequest } from "../../api/types";

interface Props {
  loading: boolean;
  onSubmit: (payload: RecommendationRequest) => void;
}

const UserSelector = ({ loading, onSubmit }: Props) => {
  const { t } = useTranslation();
  const [userId, setUserId] = useState<string>("");
  const [model, setModel] = useState<RecommendationRequest["model"]>("hybrid");
  const [topK, setTopK] = useState<number>(5);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId) return;
    // EN: When form submits, send request to recommendation API
    // FA: هنگام ارسال فرم، درخواست به API توصیه ارسال می‌شود
    onSubmit({ user_id: userId, model, top_k: topK });
  };

  return (
    <form className="card" onSubmit={handleSubmit}>
      <label className="label">{t("userId")}</label>
      <input
        className="input"
        placeholder={t("enterUserId") ?? ""}
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />

      <label className="label">{t("model")}</label>
      <select className="select" value={model} onChange={(e) => setModel(e.target.value as any)}>
        <option value="baseline">{t("baseline")}</option>
        <option value="cf">{t("cf")}</option>
        <option value="hybrid">{t("hybrid")}</option>
      </select>

      <label className="label">Top K</label>
      <input
        type="number"
        className="input"
        min={1}
        max={50}
        value={topK}
        onChange={(e) => setTopK(Number(e.target.value))}
      />

      <button className="primary-btn" type="submit" disabled={loading}>
        {loading ? "..." : t("getRecommendations")}
      </button>
    </form>
  );
};

export default UserSelector;
