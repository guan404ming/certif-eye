"use client";

import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import useReview from "@/hooks/useReview";
import {
  useMap,
  useMapsLibrary,
  Map,
  AdvancedMarker,
} from "@vis.gl/react-google-maps";
import React from "react";
import { useEffect, useState } from "react";

const Review = ({
  username,
  date,
  content,
  avatar,
}: {
  username: string;
  date: number;
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

function ScoreBadge({ score }: { score: number }) {
  if (score < 0.9) {
    if (score === -100) {
      return <Badge variant={"destructive"}>Not Found</Badge>;
    }
    return <Badge variant={"destructive"}>Fake</Badge>;
  } else if (score >= 0.9 && score <= 1.3) {
    return <Badge variant={"default"}>Medium</Badge>;
  } else {
    return <Badge variant={"success"}>Real</Badge>;
  }
}

function LocationPage({ params }: { params: { placeId: string } }) {
  const [place, setPlace] = useState<google.maps.places.PlaceResult | null>();
  const [score, setScore] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const { getPlaceScore } = useReview();

  const map = useMap();
  const placesLib = useMapsLibrary("places");
  const latlng =
    place && place.geometry?.location
      ? {
          lat: place.geometry?.location.lat(),
          lng: place.geometry?.location.lng(),
        }
      : { lat: 25.0129, lng: 121.5371 };

  useEffect(() => {
    if (!placesLib) return;
    const service = new placesLib.PlacesService(map!);
    const detailRequest = {
      placeId: params.placeId,
      language: "zh-TW",
    };

    service.getDetails(detailRequest, (placeResult, detailStatus) => {
      if (detailStatus === placesLib.PlacesServiceStatus.OK) {
        console.log(placeResult);
        setPlace(placeResult);
      }
    });

    (async () => {
      setLoading(true);
      const score = parseFloat(await getPlaceScore(params.placeId));
      setLoading(false);
      setScore(parseFloat(score.toFixed(4)));
    })();
  }, [map, placesLib, params.placeId]);

  return (
    <Card className="text-center min-w-[300px] min-h-full">
      {loading && <CardContent>Loading...</CardContent>}
      <CardContent className="flex-col space-y-4 py-4">
        <div className=" flex justify-between items-center">
          <div className="text-left space-y-1">
            <h1 className="text-lg font-bold">{place?.name}</h1>

            <div className="space-x-2">
              <Badge variant={"secondary"}>Score: {score}</Badge>
              {place?.price_level && (
                <Badge variant={"secondary"}>
                  {"$".repeat(place?.price_level ?? 0)}
                </Badge>
              )}
            </div>
          </div>

          <div className="border text-center p-2">
            <p className="font-bold">{score}</p>
            <ScoreBadge score={score}></ScoreBadge>
          </div>
        </div>

        <Map
          center={latlng}
          defaultZoom={15}
          mapId={"map"}
          reuseMaps
          className="w-full h-96"
        >
          <AdvancedMarker position={latlng} />
        </Map>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {place?.reviews &&
            place?.reviews.map((review, index) => (
              <Review
                key={index}
                username={review.author_name}
                date={review.time}
                content={review.text}
                avatar={review.profile_photo_url}
              />
            ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default React.memo(LocationPage);
