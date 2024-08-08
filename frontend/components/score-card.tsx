import { Badge } from "./ui/badge";

export function ScoreBadge({ score }: { score: number }) {
  if (score < 25) {
    if (score === -1) {
      return <Badge variant={"destructive"}>Not Found</Badge>;
    }
    return <Badge variant={"destructive"}>Fake</Badge>;
  } else if (score >= 25 && score <= 75) {
    return <Badge variant={"default"}>Medium</Badge>;
  } else {
    return <Badge variant={"success"}>Real</Badge>;
  }
}
