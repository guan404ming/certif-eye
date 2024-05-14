const URL = process.env.NEXT_PUBLIC_VERCEL_URL
  ? `https://${process.env.NEXT_PUBLIC_VERCEL_URL}/api`
  : "http://localhost:3000/api";

export default function useReview() {
  async function getReviewPoint(review: string) {
    const res = await fetch(`${URL}/infer`, {
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
