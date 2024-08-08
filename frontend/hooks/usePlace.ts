export default function usePlace() {
  async function getPlaceReviewScore(review: string) {
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

  async function getPlaceInfo(placeId: string) {
    const res = await fetch("/api/get-place-info", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "place_id": placeId,
      }),
    });
    return await res.json()
  }

  return {
    getPlaceReviewScore,
    getPlaceInfo,
  }
}
