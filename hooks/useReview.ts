export default function useReview() {
  async function getReviewPoint(review: string) {
    const res = await fetch("http://localhost:3000/api/infer", {
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

  return {
    getReviewPoint,
  }
}
