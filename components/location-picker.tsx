import React, { useState } from "react";
import {
  Marker,
  InfoWindow,
  Map,
  useMap,
  useMapsLibrary,
  type MapMouseEvent,
} from "@vis.gl/react-google-maps";
import { Button } from "./ui/button";
import { useRouter } from "next/navigation";

function LocationPicker() {
  const [place, setPlace] = useState<google.maps.places.PlaceResult | null>(
    null
  );

  const map = useMap();
  const placesLib = useMapsLibrary("places");
  const router = useRouter();

  const latlng =
    place && place.geometry?.location
      ? {
          lat: place.geometry?.location.lat(),
          lng: place.geometry?.location.lng(),
        }
      : { lat: 25.0129, lng: 121.5371 };

  const handleMapClick = (e: MapMouseEvent) => {
    if (placesLib && e.detail.placeId) {
      const service = new placesLib.PlacesService(map!);
      const detailRequest = {
        placeId: e.detail.placeId,
        fields: [
          "name",
          "formatted_address",
          "geometry",
          "place_id",
          "rating",
          "type",
        ],
        language: "zh-TW",
      };

      service.getDetails(detailRequest, (placeResult, detailStatus) => {
        if (detailStatus === placesLib.PlacesServiceStatus.OK) {
          console.log(placeResult);
          setPlace(placeResult);
        }
      });
    }

    e.stop();
  };

  return (
    <>
      <Map
        defaultCenter={{ lat: 25.0129, lng: 121.5371 }}
        defaultZoom={15}
        onClick={handleMapClick}
        mapId={process.env.NEXT_PUBLIC_MAP_ID}
        reuseMaps
        className="w-full h-96"
      >
        <Marker position={latlng} />
        {place && (
          <InfoWindow
            position={latlng}
            onCloseClick={() => setPlace(null)}
            pixelOffset={[0, -40]}
            headerDisabled
          >
            <h1 className="text-lg font-bold">{place.name}</h1>
            <p>{place.formatted_address}</p>
            <Button
              className="h-5 m-2 w-20"
              onClick={() => router.push(`/${place.place_id}`)}
            >
              Analysis
            </Button>
          </InfoWindow>
        )}
      </Map>
    </>
  );
}

export default React.memo(LocationPicker);
