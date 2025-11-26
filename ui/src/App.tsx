import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import AppLayout from "./components/layout/AppLayout";
import UserSelector from "./components/recommendations/UserSelector";
import RecommendationList from "./components/recommendations/RecommendationList";
import ItemDetailsDrawer from "./components/items/ItemDetailsDrawer";
import { fetchHealth, fetchRecommendations, fetchSimilarItems, fetchItem } from "./api/client";
import { ItemMetadata, RecommendationRequest, RecommendationResponse, SimilarItemsResponse } from "./api/types";

function App() {
  const { i18n, t } = useTranslation();
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const [similarItems, setSimilarItems] = useState<SimilarItemsResponse | null>(null);
  const [selectedItem, setSelectedItem] = useState<ItemMetadata | null>(null);

  // EN: Ping backend health on load to update status indicator
  // FA: در شروع برنامه وضعیت سلامت بک‌اند را بررسی می‌کنیم
  useEffect(() => {
    fetchHealth()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  // EN: Helper to change document direction based on active language
  // FA: کمک برای تعیین جهت متن بر اساس زبان فعال
  useEffect(() => {
    document.documentElement.lang = i18n.language;
    document.documentElement.dir = i18n.language === "fa" ? "rtl" : "ltr";
  }, [i18n.language]);

  const handleRecommend = async (payload: RecommendationRequest) => {
    setLoading(true);
    setError(null);
    setSimilarItems(null);
    setSelectedItem(null);
    try {
      const res = await fetchRecommendations(payload);
      setRecommendations(res);
    } catch (err: any) {
      console.error("recommendations error:", err);
      setError(err.message || "Failed to fetch recommendations");
    } finally {
      setLoading(false);
    }
  };

  const handleViewSimilar = async (itemId: string, top_k: number = 5) => {
    setLoading(true);
    setError(null);
    try {
      const [sims, meta] = await Promise.all([
        fetchSimilarItems({ item_id: itemId, top_k }),
        fetchItem(itemId),
      ]);
      setSimilarItems(sims);
      setSelectedItem(meta);
    } catch (err: any) {
      console.error("similar-items error:", err);
      setError(err.message || "Failed to fetch similar items");
    } finally {
      setLoading(false);
    }
  };

  const clearDrawer = () => {
    setSimilarItems(null);
    setSelectedItem(null);
  };

  const recItems = useMemo(() => recommendations?.items ?? [], [recommendations]);

  return (
    <AppLayout backendOnline={backendOnline}>
      <div className="control-panel">
        <UserSelector loading={loading} onSubmit={handleRecommend} />
        {error && <div className="error-banner">{error}</div>}
      </div>
      <div className="content-panel">
        <h2 className="section-title">{t("recommendedItems")}</h2>
        <RecommendationList items={recItems} onViewSimilar={(id) => handleViewSimilar(id, 6)} loading={loading} />
        {recItems.length === 0 && !loading && (
          <div className="empty-state">{t("noResults")}</div>
        )}
      </div>
      <ItemDetailsDrawer
        open={!!similarItems}
        onClose={clearDrawer}
        anchorItem={selectedItem}
        similarItems={similarItems?.items ?? []}
      />
    </AppLayout>
  );
}

export default App;
