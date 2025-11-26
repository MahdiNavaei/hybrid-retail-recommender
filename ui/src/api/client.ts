import axios from "axios";
import {
  HealthResponse,
  RecommendationRequest,
  RecommendationResponse,
  SimilarItemsRequest,
  SimilarItemsResponse,
  ItemMetadata,
} from "./types";

// EN: Axios instance with base URL from env (falls back to localhost)
// FA: نمونه Axios با آدرس پایه از متغیر محیطی (در صورت نبود، لوکال)
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
  timeout: 10000,
});

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await apiClient.get<HealthResponse>("/health");
  return res.data;
}

export async function fetchRecommendations(
  payload: RecommendationRequest
): Promise<RecommendationResponse> {
  try {
    // EN: Request recommendations for the given user/model
    // FA: درخواست توصیه برای کاربر/مدل مشخص
    const res = await apiClient.post<RecommendationResponse>("/recommendations", payload);
    return res.data;
  } catch (err: any) {
    // EN: Map Axios error to a readable message (e.g., 404 user not found)
    // FA: خطای Axios را به پیام خوانا (مثلاً ۴۰۴ کاربر یافت نشد) تبدیل می‌کنیم
    const msg = err?.response?.data?.detail || err.message || "Request failed";
    throw new Error(msg);
  }
}

export async function fetchSimilarItems(
  payload: SimilarItemsRequest
): Promise<SimilarItemsResponse> {
  try {
    const res = await apiClient.post<SimilarItemsResponse>("/similar-items", payload);
    return res.data;
  } catch (err: any) {
    const msg = err?.response?.data?.detail || err.message || "Request failed";
    throw new Error(msg);
  }
}

export async function fetchItem(itemId: string): Promise<ItemMetadata> {
  try {
    const res = await apiClient.get<ItemMetadata>(`/items/${itemId}`);
    return res.data;
  } catch (err: any) {
    const msg = err?.response?.data?.detail || err.message || "Request failed";
    throw new Error(msg);
  }
}
