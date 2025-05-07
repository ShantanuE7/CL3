import java.rmi.*;

public class HotelClient
{
    public static void main(String[] args)
    {
        try {
            HotelInterface hotel = (HotelInterface) Naming.lookup("rmi://localhost/HotelService");

            System.out.println(hotel.bookRoom("John Doe"));
            System.out.println(hotel.bookRoom("Alice"));
            System.out.println(hotel.cancelBooking("John Doe"));
            System.out.println(hotel.cancelBooking("Bob")); // Not booked

        }
        catch (Exception e)
        {
            System.out.println("Client Error: " + e);
        }
    }
}
