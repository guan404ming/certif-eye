"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import useReview from "@/hooks/useReview";
import Image from "next/image";
import {
  useMap,
  useMapsLibrary,
  Map,
  AdvancedMarker,
} from "@vis.gl/react-google-maps";
import React from "react";
import { useEffect, useState } from "react";
import { ReviewCard } from "@/components/review-card";
import { ScoreBadge } from "@/components/score-card";

function LocationPage({ params }: { params: { placeId: string } }) {
  const [place, setPlace] = useState<google.maps.places.PlaceResult | null>();
  const [score, setScore] = useState<number>(0);
  const [wordCloud, setWordCloud] = useState<any>();

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
      const res = await getPlaceScore(params.placeId);
      const score = parseFloat(res.score);
      setLoading(false);
      setScore(parseFloat(score.toFixed(2)));
      setWordCloud(res.wordcloud);
    })();
  }, [map, placesLib, params.placeId]);

  return (
    <Card className="text-center min-w-[300px] min-h-full">
      <CardContent className={"flex-col space-y-4 py-4"}>
        <div className=" flex justify-between items-center">
          <div className="text-left space-y-1">
            <h1 className="text-lg font-bold">{place?.name}</h1>

            <div className="space-x-2">
              {place?.rating && (
                <Badge variant={"secondary"}>{place?.rating}</Badge>
              )}
              {place?.price_level && (
                <Badge variant={"secondary"}>
                  {"$".repeat(place?.price_level ?? 0)}
                </Badge>
              )}
            </div>
          </div>

          <div className="border text-center p-2">
            {!loading ? (
              <>
                <p className="font-bold">{score}</p>
                <ScoreBadge score={score}></ScoreBadge>
              </>
            ) : (
              <div className="flex items-center space-x-4">
                <div className="space-y-2">
                  <Skeleton className="h-4 w-[80px]" />
                  <Skeleton className="h-4 w-[60px]" />
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Map
            center={latlng}
            defaultZoom={15}
            mapId={"map"}
            reuseMaps
            className="w-full h-96 md:h-auto md:min-h-[200px] "
          >
            <AdvancedMarker position={latlng} />
          </Map>

          {wordCloud && (
            <div className="flex justify-center items-center">
              <Image
                src={`data:image/png;base64,${wordCloud}`}
                alt="upload pic"
                width={200}
                height={200}
                className=" h-full w-auto"
              />
            </div>
          )}
        </div>

        <h1 className="font-bold text-xl text-left">Reviews</h1>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {place?.reviews &&
            place?.reviews.map((review, index) => (
              <ReviewCard
                key={index}
                username={review.author_name}
                date={review.relative_time_description}
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
