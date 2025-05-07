import java.rmi.*;
import java.rmi.server.*;
import java.rmi.registry.*;
import java.util.*;

public class HotelServer extends UnicastRemoteObject implements HotelInterface
{
    private static final long serialVersionUID = 1L;
    private Map<String, String> bookings;

    protected HotelServer() throws RemoteException
    {
        bookings = new HashMap<>();
    }

    public synchronized String bookRoom(String guestName) throws RemoteException
    {
        if (bookings.containsKey(guestName)) {
            return "Guest " + guestName + " already has a booking.";
        }
        bookings.put(guestName, "Room Booked");
        return "Room successfully booked for " + guestName;
    }

    public synchronized String cancelBooking(String guestName) throws RemoteException
    {
        if (!bookings.containsKey(guestName))
        {
            return "No booking found for guest " + guestName;
        }
        bookings.remove(guestName);
        return "Booking cancelled for " + guestName;
    }

    public static void main(String[] args)
    {
        try
        {
            LocateRegistry.createRegistry(1099); // start RMI registry on port 1099
            HotelServer server = new HotelServer();
            Naming.rebind("HotelService", server);
            System.out.println("Hotel Booking Server is ready.");
        }
        catch (Exception e)
        {
            System.out.println("Server Error: " + e);
        }
    }
}
