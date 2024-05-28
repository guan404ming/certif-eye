"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import useReview from "@/hooks/useReview";
import { useRef, useState } from "react";

export default function Home() {
  const [score, setScore] = useState<number>(0);
  const { getReviewPoint } = useReview();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const ScoreBadge = ({ score }: { score: number }) => {
    if (score < 0.9) {
      return <Badge variant={"destructive"}>Fake</Badge>;
    } else if (score >= 0.9 && score <= 1.3) {
      return <Badge variant={"default"}>Medium</Badge>;
    } else {
      return <Badge variant={"success"}>Real</Badge>;
    }
  };

  return (
    <Card className="text-center w-1/2 min-w-[300px] min-h-full">
      <CardHeader className=" text-xl font-bold">👀 Certif-Eye 👀</CardHeader>
      <CardContent className=" flex-col space-y-4">
        <Textarea ref={textareaRef} placeholder="Enter some review......" />
        <Button
          className="w-full"
          onClick={async () => {
            console.log(textareaRef.current?.value);
            const score = parseFloat(
              await getReviewPoint(textareaRef.current?.value!)
            );
            setScore(parseFloat(score.toFixed(4)));
          }}
        >
          Enter
        </Button>
        {score !== 0 && (
          <>
            <Separator />
            <CardContent className="flex items-center justify-center space-x-2 pb-0">
              <p className="text-lg font-semibold">Score: {score}</p>
              <ScoreBadge score={score}></ScoreBadge>
            </CardContent>
          </>
        )}
      </CardContent>
    </Card>
  );
}
