export default function useReview() {
  async function getReviewPoint(review: string) {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data: review,
      }),
    });
    return await res.text()
  }

  async function getPlaceScore(placeId: string) {
    const res = await fetch("/api/get-score", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "place_id": placeId,
      }),
    });
    return await res.text()
  }

  return {
    getReviewPoint,
    getPlaceScore,
  }
}
