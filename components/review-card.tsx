
import { Avatar, AvatarImage } from "./ui/avatar";
import { Card, CardHeader, CardContent } from "./ui/card";

export const ReviewCard = ({
    username,
    date,
    content,
    avatar,
  }: {
    username: string;
    date: string;
    content: string;
    avatar: string;
  }) => {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="text-left">
              <p className="font-bold">{username}</p>
              <p className="text-sm text-gray-500">{date}</p>
            </div>
            <Avatar>
              <AvatarImage src={avatar}></AvatarImage>
            </Avatar>
          </div>
        </CardHeader>
        <CardContent className=" text-left">{content}</CardContent>
      </Card>
    );
  };