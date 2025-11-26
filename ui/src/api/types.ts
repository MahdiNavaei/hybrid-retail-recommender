// EN: Shared TypeScript types for API payloads
// FA: انواع تایپ‌اسکریپت مشترک برای payload های API

export interface ItemMetadata {
  item_id: string;
  title?: string | null;
  category?: string | null;
  score?: number | null;
  attributes?: { name: string; value: string }[];
}

export interface RecommendationRequest {
  user_id: string;
  top_k?: number;
  model?: "baseline" | "cf" | "hybrid";
}

export interface RecommendationResponse {
  user_id: string;
  model: string;
  top_k: number;
  items: ItemMetadata[];
}

export interface SimilarItemsRequest {
  item_id: string;
  top_k?: number;
}

export interface SimilarItemsResponse {
  item_id: string;
  top_k: number;
  items: ItemMetadata[];
}

export interface HealthResponse {
  status: string;
  detail: string;
}
