(* Business Logic *)
fun createCO(id: int) = 
let
	val co_id = "co"^Int.toString id
	val hus = round(uniform(5.5, 30.5))
	val goods = hus*50
in
	(co_id, goods)
end;

fun expb_delay(m, mn, mx) = ModelTime.fromInt(round(expb_st(m, mn, mx)))
fun normal_delay(m, std) = ModelTime.fromInt(round(norm_st(m, std)))

fun CODelay() = ModelTime.fromInt(round(norm_r_at_delay(week/30.0)))
fun RegisterCODelay() = expb_delay(2.0*day, 30.0*minute, 5.0*day)
fun CreateTDDelay() = expb_delay(1.0*day, 30.0*minute, 2.0*day)
fun BookVHDelay() = expb_delay(1.0*hour, 5.0*minute, 2.0*hour)
fun OrderCRDelay() = expb_delay(3.0*hour, 10.0*minute, 5.0*hour)
fun CRToPlantDelay() = 
let 
	val b = uniform(0.0,1.0)
in
	if b < 0.85 
	then normal_delay(20.0*minute, 3.0*minute) 
	else normal_delay(75.0*minute, 10.0*minute)
end;

fun CollectGoodsDelay() = expb_delay(3.0*minute, 30.0, 8.0*minute)
fun LoadTruckDelay() = expb_delay(3.0*minute, 30.0, 5.0*minute)
fun DriveToTerminalDelay() = 
let 
	val b = uniform(0.0,1.0)
in
	if b < 0.6
	then normal_delay(10.0*minute, 2.0*minute)
	else normal_delay(40.0*minute, 5.0*minute)
end;
fun WeighDelay() = expb_delay(3.0*minute, 30.0, 6.0*minute)
fun WeighToBayDelay() = expb_delay(5.0*minute, 60.0, 8.0*minute)
fun WeighToStockDelay() = expb_delay(3.0*minute, 30.0, 5.0*minute)
fun StockToBayDelay() = expb_delay(5.0*minute, 1.0*minute, 5.0*minute)
fun LoadToVehicleDelay() = expb_delay(5.0*minute, 30.0, 8.0*minute)

fun initialCOTime() = 
let
	val co_time_real = process_start_time()
	val co_time_clock = co_time_real - preprocess_start_time()
in
	ModelTime.fromInt(round(co_time_clock))
end;


fun newCRs(0, _, _, _) = [] | newCRs(nof_crs: INT, nof_hus: INT, td_id, id: INT) =
let 
	val cr_id = "cr"^(Int.toString id)
	val veh_id = "NONE"
	val hus = []
	val status = empt
	val cr_weight = ~1.0
	val rnd = uniform(0.0,1.0)
	val cr_prio = if rnd < 0.1 then crp_high else crp_normal
	val nof_hus_cr = if nof_crs = 1 then nof_hus else 6
	val cr = (cr_id, td_id, veh_id, nof_hus_cr, [], status, cr_weight, cr_prio)
in
	cr::newCRs(nof_crs - 1, nof_hus - 6, td_id, id + 1)
end;
fun TDForCO((oid, goods): CustomerOrder, id1: INT, id2: INT) =
let
	val td_id = "td"^Int.toString id1
	val nof_hus = goods div 50 
	val nof_crs = ((nof_hus - 1) div 6) + 1
	val crs = newCRs(nof_crs, nof_hus, td_id, id2)
in
	((td_id, crs), id2 + nof_crs)
end;

fun HUsForCR_recursive(cr_id, 0, hus, id) = (hus, id) | HUsForCR_recursive(cr_id, nof_hus: INT, hus, id) =
let
	val hu = ("hu"^Int.toString id, cr_id)
in 
	HUsForCR_recursive(cr_id, nof_hus - 1, hu::hus, id + 1)
end;
fun HUsForCR((cr_id, _, _, nof_hus, _, _, _, _): Container, id: INT) = HUsForCR_recursive(cr_id, nof_hus, [], id);

fun deleteCR([], cr: Container) = [] | deleteCR(cr2::crs: Containers, cr: Container) = if #1 cr = #1 cr2 then crs else cr2::deleteCR(crs, cr)
fun decrPendingCRs((td_id, crs): TransportDocument, cr: Container) = (td_id, deleteCR(crs, cr))

fun departSoon(tm) =  ModelTime.add(tm, ModelTime.fromInt(60*60*24)) > time()

fun generateVH((_, clock): VHTime, id: int) = 
let
	val vh_id = "vh"^Int.toString id
	val idle_cap = round(uniform(0.0-0.5, 150.5))
in
	(vh_id, idle_cap, [], [], clock)
end;

fun initialVHTime() = 
let
	val vh_time_real = tuesday_may_30_2023()
	val vh_time_clock = vh_time_real - preprocess_start_time()
in
	(Tuesday, vh_time_clock)
end;

fun nextVHTime((weekday, clock): VHTime) = 
let
	val delta = if weekday = Tuesday then 3.0*day else 4.0*day
	val next_weekday = if weekday = Friday then Tuesday else Friday
	val next_time_0 = clock + delta
	(* summer / winter time mess handling *)
	val next_time = 
		if Date.hour(t2date(next_time_0)) = 10 then next_time_0 + hour else 
		if Date.hour(t2date(next_time_0)) = 12 then next_time_0 - hour else next_time_0
in
	(next_weekday, next_time)
end;


(* ***********************************************)
(* scheduling containers to vehicles logic *)
(* ***********************************************)

(* this is to catch the case that there is no vehicle with enough capacity for the order
	standard ml seems to not provide a proper exception handling, so identify this error through postprocessing (in the .ipynb) *)
fun ERROR_VEHICLE() = ("errorVH", 100000, [], [], now()+now())

fun assign_containers_to_vehicle([], scheduled_crs, cr_departure_times, assigned_veh) = (scheduled_crs, cr_departure_times, assigned_veh) |
	assign_containers_to_vehicle((cr_id, td_id, _, nofHus, hus, status, weight, prio)::crs: Containers, scheduled_crs, cr_departure_times, veh) 
	= let 
		val (veh_id, idle_cap, scheduled_cr_ids, loaded_crs, clock) = veh
		val scheduled_cr = (cr_id, td_id, veh_id, nofHus, hus, status, weight, prio)
		val cr_departure_time = (cr_id, clock)
		val assigned_veh = (veh_id, idle_cap - 1, cr_id::scheduled_cr_ids, loaded_crs, clock)
	in
		assign_containers_to_vehicle(crs, scheduled_cr::scheduled_crs, cr_departure_time::cr_departure_times, assigned_veh)
	end;

fun assign_vehicle_prio_high(crs, [], checked: Vehicles) 
= let
	(* error: no more vehicle available *)
	val veh = ERROR_VEHICLE();
	val (scheduled_crs, cr_departure_times, assigned_veh) = assign_containers_to_vehicle(crs, [], [], veh)
in
	(scheduled_crs, cr_departure_times, checked^^[assigned_veh])
end | assign_vehicle_prio_high(crs, veh::unchecked, checked)
(* greedy: take next fitting vehicle *)
= let
	val (vh_id, idle_cap, scheduled_cr_ids, loaded_crs, departure) = veh
in 
	if (idle_cap >= List.length crs) andalso (departure - now() >= 1.2*day)
	then let
			val (scheduled_crs, cr_departure_times, assigned_veh) = assign_containers_to_vehicle(crs, [], [], veh)
		in
			(scheduled_crs, cr_departure_times, checked^^[assigned_veh]^^unchecked)
		end
	else assign_vehicle_prio_high(crs, unchecked, checked^^[veh] )
end;

fun get_vehicle_idle_capacities([]) = [] | get_vehicle_idle_capacities((_, idle_cap, _, _, _)::vehs) = idle_cap::get_vehicle_idle_capacities(vehs);

fun assign_vehicle_prio_normal(crs: Containers, vehs: Vehicles) 
= let
	val idle_capacities = get_vehicle_idle_capacities(vehs)
	val b = uniform(0.0, 1.0)
	(* assumption: there are always (at least) 8 vehicles *)
	val n = if b < 0.05 then 1 else
			if b < 0.20 then 2 else
			if b < 0.40 then 3 else
			if b < 0.60 then 4 else
			if b < 0.80 then 5 else
			if b < 0.90 then 6 else 7
	val veh = List.nth(vehs, n)
in 
	if List.nth(idle_capacities, n) >= List.length crs
	then let
		val (scheduled_crs, cr_deparutre_times, assigned_veh) = assign_containers_to_vehicle(crs, [], [], veh)
		val earlier_vehs = List.take(vehs, n)
		val later_vehs = List.drop(vehs, n+1)
	in
		(scheduled_crs, cr_deparutre_times, earlier_vehs^^[assigned_veh]^^later_vehs)
	end
	(* note: this is not safe especially if no vehicle has enough idle capacity *)
	else assign_vehicle_prio_normal(crs, vehs)
end;

fun split_crs_by_prio_recursively(highs, normals, []) = (highs, normals) |
	split_crs_by_prio_recursively(highs, normals, cr::crs: Containers) =
	let
		val (cr_id, td_id, veh_id, nof_hus_cr, hus, status, cr_weight, cr_prio) = cr
	in
		if cr_prio = crp_high 
		then split_crs_by_prio_recursively(cr::highs, normals, crs)
		else split_crs_by_prio_recursively(highs, cr::normals, crs)
	end;

fun split_crs_by_prio(crs) = split_crs_by_prio_recursively([], [], crs)

fun bookVehicles((td_id, crs): TransportDocument, vehs: Vehicles)
= let 
	(* 10% of the containers have high prio and are shipped jointly as soon as possible *)
	(* 90% are jointly assigned to some vehicle over the next few weeks *)
	val (high_prio_crs, normal_prio_crs) = split_crs_by_prio(crs)
	val vehicles_idle_caps = get_vehicle_idle_capacities(vehs)
	val (high_prio_crs, high_prio_dep_times, vehs) = assign_vehicle_prio_high(high_prio_crs, vehs, [])
	val (normal_prio_crs, normal_prio_dep_times, vehs) = assign_vehicle_prio_normal(normal_prio_crs, vehs)
	val crs = high_prio_crs^^normal_prio_crs
	val td: TransportDocument = (td_id, crs)
	val crdts = high_prio_dep_times^^normal_prio_dep_times
in
	(td, vehs, crdts)
end;

fun untilAboutOneDayBeforeDepartureTime((_, _, _, _, clock): Vehicle) = 
let 
	val slightly_more_than_one_day = clock - now() - 1.5*day
	val sample_in_days = normal(slightly_more_than_one_day / day, 0.05)
	val sample_in_seconds = round(sample_in_days * day)
in 
	ModelTime.fromInt(sample_in_seconds)
end;
fun untilDepartureTime((_, _, _, _, clock): Vehicle) = ModelTime.fromInt(round(clock - now()))


(* Container handling should be initiated at least three days before vehicle departure *)
fun samplePickTime(departure: Clock) = 
let
	val nowtime = now()
	val maxtime = departure - 3.0*day - nowtime
	val halftime = maxtime/2.0
	val depart_soon = maxtime < 0.0
in
	if depart_soon then ModelTime.fromInt(0)
	else let
			val b = uniform(0.0,1.0)
			val t = if b < 0.8 then uniform(0.0, halftime) else uniform(halftime, maxtime)
		in
			ModelTime.fromInt(round(t))
		end
end;